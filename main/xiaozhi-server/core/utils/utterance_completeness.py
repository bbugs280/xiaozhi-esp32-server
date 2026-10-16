"""Utterance completeness gate.

## Why this exists

The server's VAD closes a turn on *silence*. But a thinking pause is physically
identical to finishing a sentence — both are silence — so VAD cannot tell "I'm
done" from "I'm still thinking". Upstream, every VAD boundary was therefore
treated as a turn boundary and the LLM was fired immediately, answering
half-sentences.

Observed 2026-10-11 11:18–11:20 (one thought, three LLM turns and three replies):

    11:18:44  识别文本: 一個關於天頭痛嘅。        -> LLM answered
    11:19:06  识别文本: 問題啦，因為我呢排成日都有偏頭痛，咁，我想睇下啲咩方案，所以。  -> LLM answered
    11:19:44  识别文本: 我想話加個。             -> LLM answered "你想我幫你加個咩？"

Each fragment ends on a particle or bare classifier (嘅 / 所以 / 加個) — text that
*cannot* end a Chinese sentence. The server had the text in hand and never looked.

## The gate

Sit between ASR and the LLM. Ask "is this a finished thought?" before spending a
turn on it:

    incomplete  -> HOLD the text, don't fire the LLM; merge with the next fragment
    complete    -> answer now (unchanged behaviour)

Two tiers, cheapest first:

1. **Trailing-token pre-filter** (free, no network). A fragment ending on a
   particle / conjunction / bare classifier is syntactically incapable of being a
   complete thought -> INCOMPLETE without an LLM call. This keeps the common
   "clean full sentence" case at *zero* added latency.
2. **LLM judge** (only for text the pre-filter can't decide). Measured ~1.7s on
   the local `voice-satellite` endpoint, 5/5 accurate on the real fragments above.

The LLM judge is FAIL-OPEN: any error/timeout treats the utterance as complete, so
a broken judge degrades to today's behaviour rather than wedging the conversation.
"""

from typing import Optional

# Fragments ending on these cannot be a complete Chinese/Cantonese sentence.
# Deliberately conservative: only unambiguous non-terminal tokens, so a clean
# full sentence never pays for an LLM call.
_TRAILING_INCOMPLETE_TOKENS = (
    # possessive / structural particles — nothing can end a sentence on these
    "嘅", "的", "地", "得", "之",
    # conjunctions / connectives that demand a continuation
    "所以", "因為", "同埋", "但係", "不過", "而且", "然之後", "跟住",
    "如果", "雖然", "就算", "的話", "既", "係咪", "定係", "或者", "例如",
    # subordinators / fillers that trail off
    "咁", "嗰個", "呢個", "一個", "喺", "同", "而", "就", "都", "仲", "又",
    # bare classifiers (cannot stand alone: 加個 / 買個 / 睇個)
    "個", "隻", "件", "條", "張", "杯", "份", "種",
)

# Punctuation we strip before inspecting the tail.
_TRAILING_PUNCT = "。！？!?.,，、…～~ \t\n\r\"'“”‘’「」『』（）()"

# Verdicts
COMPLETE = "complete"
INCOMPLETE = "incomplete"


def has_trailing_incomplete_token(text: str) -> bool:
    """True if `text` ends on a token that cannot terminate a sentence.

    Pure, synchronous, network-free — this is tier 1 of the gate.
    """
    if not text:
        return False
    stripped = text.strip().rstrip(_TRAILING_PUNCT)
    if not stripped:
        return False
    return stripped.endswith(_TRAILING_INCOMPLETE_TOKENS)


_LLM_SYSTEM_PROMPT = (
    "你係語音助理嘅斷句判斷器。用戶講嘅係粵語口語(可能混英文)。"
    "判斷呢句係咪一個完整嘅意思，定係講到一半停咗(仲未講完，等緊繼續)。"
    '輸出必須係單一 JSON，唔好有其他字：{"complete": true} 或 {"complete": false}'
)


def _parse_complete(raw: str) -> Optional[bool]:
    """Extract the boolean from the judge's reply. None if unparseable."""
    if not raw:
        return None
    import json
    import re

    text = raw.strip()
    # Strip a ```json fence if the model added one.
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "complete" in payload:
            return bool(payload["complete"])
    except (ValueError, TypeError):
        pass
    # Lenient fallback for prose answers ("呢句未講完" / "false").
    lowered = text.lower()
    if "false" in lowered:
        return False
    if "true" in lowered:
        return True
    return None


_JUDGE_TIMEOUT_S = 8.0


async def judge_completeness_llm(conn, text: str) -> Optional[bool]:
    """Tier 2: ask the LLM whether `text` is a finished thought.

    Returns True/False, or None when the judge could not be reached/parsed/
    timed out (callers must treat None as "complete" — fail open, preserve
    today's behaviour rather than wedging the turn).
    """
    import asyncio

    llm = getattr(conn, "llm", None)
    if llm is None:
        return None

    dialogue = [
        {"role": "system", "content": _LLM_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    try:
        # ⚠️ The adapter interface is `response(session_id, dialogue)` — see
        # LLMProviderBase. Calling it as `response(messages)` raises
        # `TypeError: missing 1 required positional argument: 'dialogue'`,
        # which the handler below swallows as a fail-open. That silently
        # disabled tier 2 entirely. Argument shape is load-bearing.
        raw = llm.response("", dialogue)
        if hasattr(raw, "__await__"):
            # ⚠️ Bounded wait: the fail-open contract promises we degrade to
            # "complete" on a *hang* as well as on an error. A bare `await`
            # would block the ASR turn indefinitely if the endpoint stalls.
            raw = await asyncio.wait_for(raw, timeout=_JUDGE_TIMEOUT_S)
        # ⚠️ Adapters return a GENERATOR of text chunks, not a string.
        # `str(<generator>)` is "<generator object ...>", which parses as
        # unparseable and fails open — so the verdict must be JOINED.
        result = await asyncio.to_thread(_consume_llm_result, raw)
    except Exception as exc:  # noqa: BLE001 — must never break the turn
        try:
            conn.logger.bind(tag=__name__).warning(
                f"completeness judge failed ({type(exc).__name__}: {exc}) — failing open"
            )
        except Exception:  # noqa: BLE001
            pass
        return None

    return _parse_complete(result)


def _consume_llm_result(raw) -> str:
    """Normalise whatever the adapter returned into the judge's reply text.

    Handles the three shapes the adapters actually produce:
      - a generator/iterable of text chunks (the common case) -> joined
      - a dict (`{"content": ...}` / `{"text": ...}`)
      - a plain string

    ⚠️ Iterating the generator is blocking work; callers must run this via
    `asyncio.to_thread` so a slow stream cannot stall the event loop.
    """
    if raw is None:
        return ""
    if isinstance(raw, dict):
        return str(raw.get("content") or raw.get("text") or "")
    if isinstance(raw, str):
        return raw
    if hasattr(raw, "__iter__") or hasattr(raw, "__next__"):
        return "".join(str(chunk or "") for chunk in raw)
    return str(raw)


async def is_utterance_complete(conn, text: str) -> bool:
    """Full gate: cheap pre-filter first, LLM judge only when needed.

    Fail-open throughout: anything unexpected resolves to "complete", i.e. the
    behaviour the app had before this gate existed.
    """
    if not text or not text.strip():
        return True
    # Tier 1 — free. A trailing non-terminal token is decisive: no LLM needed.
    if has_trailing_incomplete_token(text):
        return False
    # Tier 2 — paid. Only reached by text the cheap filter can't settle.
    verdict = await judge_completeness_llm(conn, text)
    return True if verdict is None else verdict
