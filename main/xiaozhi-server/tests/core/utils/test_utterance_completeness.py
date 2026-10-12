"""Regression tests for the utterance completeness gate.

## The bug these guard (2026-10-11)

Vincent: *"I'm trying to interrupt or continue speaking while it is still in the
thinking stage, but instead it breaks my questions into a couple of stages in
batches to answer."*

Server log, one thought spoken with thinking pauses — THREE LLM turns, three
replies:

    11:18:44  识别文本: 一個關於天頭痛嘅。                      -> LLM answered
    11:19:06  识别文本: 問題啦，因為我呢排成日都有偏頭痛，咁，我想睇下啲咩方案，所以。  -> LLM answered
    11:19:44  识别文本: 我想話加個。                           -> LLM answered "你想我幫你加個咩？"

Each fragment ends on 嘅 / 所以 / 加個 — text that cannot terminate a Chinese
sentence. The server's VAD treats every silence as end-of-turn and fires the LLM
immediately, so a half-sentence gets a full answer.

⚠️ EVENT ORDER MATTERS (this repo has been burned by tests that passed on buggy
code): the fragment must reach the gate as a *completed ASR result* — i.e. after
the VAD has already closed the turn — because that is exactly the state the bug
occurs in. Do not fire the gate before the ASR text exists.
"""

import asyncio
from unittest import mock

import pytest

from core.utils import utterance_completeness as uc


class _StubConn:
    """Minimal ConnectionHandler stand-in for the gate's needs."""

    def __init__(self, llm=None):
        self.llm = llm
        self.logger = _FakeLogger()


class _FakeLogger:
    def bind(self, **kwargs):
        return self

    def info(self, msg):
        pass

    def warning(self, msg):
        pass


class _FakeLLM:
    """Records calls and returns a scripted verdict."""

    def __init__(self, verdict_json):
        self.verdict_json = verdict_json
        self.calls = []

    def response(self, messages):
        self.calls.append(messages)
        return self.verdict_json


# --------------------------------------------------------------------------
# Tier 1 — trailing-token pre-filter (free, network-free)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "我想話加個。",              # ends on bare classifier 個
        "一個關於天頭痛嘅。",          # ends on possessive particle 嘅
        "問題啦，因為我呢排成日都有偏頭痛，咁，我想睇下啲咩方案，所以。",  # ends on 所以
        "我想話加個",                # no trailing punctuation
        "我買張",                    # bare classifier 張
    ],
)
def test_tier1_flags_real_incomplete_fragments(text):
    """The exact fragments that produced the 3-batch cascade must be
    flaggable without an LLM call."""
    assert uc.has_trailing_incomplete_token(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "我想問個胸波同之前有咩唔同啊？",
        "講翻今日新聞啦。",
        "今日天氣今日天氣。",
        "我唔使啦，講下單啦。",
    ],
)
def test_tier1_does_not_overblock_complete_sentences(text):
    """A clean full sentence must fall through to tier 2 — NOT be called
    incomplete. Over-blocking here would add latency to every normal turn."""
    assert uc.has_trailing_incomplete_token(text) is False


# --------------------------------------------------------------------------
# Gate integration — an incomplete fragment must NOT reach the LLM answer path
# --------------------------------------------------------------------------

def test_incomplete_fragment_is_held_not_answered():
    """RED for the user's symptom: `我想話加個。` must be judged incomplete, so
    the caller holds it instead of spending an answer turn on it."""
    conn = _StubConn(llm=_FakeLLM('{"complete": true}'))

    verdict = asyncio.run(uc.is_utterance_complete(conn, "我想話加個。"))

    assert verdict is False, (
        "an incomplete fragment was judged COMPLETE — the LLM will answer a "
        "half-sentence, reproducing the 3-batch cascade"
    )
    # Tier 1 settled it; the LLM must not have been consulted.
    assert conn.llm.calls == [], "tier-1 hit should not cost an LLM call"


def test_complete_sentence_is_answered():
    """The complementary invariant: a genuine full question must pass the gate,
    or the fix would wedge every normal turn."""
    conn = _StubConn(llm=_FakeLLM('{"complete": true}'))

    verdict = asyncio.run(uc.is_utterance_complete(conn, "我想問個胸波同之前有咩唔同啊？"))

    assert verdict is True


# --------------------------------------------------------------------------
# Tier 2 — LLM judge behaviour
# --------------------------------------------------------------------------

def test_tier2_llm_judge_can_mark_incomplete():
    """When tier 1 can't decide, the LLM's `false` verdict must hold the turn."""
    conn = _StubConn(llm=_FakeLLM('{"complete": false}'))

    verdict = asyncio.run(uc.is_utterance_complete(conn, "空炮"))

    assert verdict is False
    assert len(conn.llm.calls) == 1, "tier 2 should consult the LLM exactly once"


def test_tier2_parses_prose_verdict():
    """The judge may answer in prose instead of JSON — must still parse."""
    conn = _StubConn(llm=_FakeLLM("呢句未講完，complete: false"))

    assert asyncio.run(uc.is_utterance_complete(conn, "空炮")) is False


def test_tier2_handles_json_fence():
    conn = _StubConn(llm=_FakeLLM('```json\n{"complete": false}\n```'))

    assert asyncio.run(uc.is_utterance_complete(conn, "空炮")) is False


# --------------------------------------------------------------------------
# Fail-open invariants — a broken judge must NOT wedge the conversation
# --------------------------------------------------------------------------

def test_judge_failure_fails_open():
    """If the LLM errors, treat the utterance as complete → degrade to today's
    behaviour rather than dropping the user's turn silently."""

    class _ExplodingLLM:
        def response(self, messages):
            raise RuntimeError("endpoint down")

    conn = _StubConn(llm=_ExplodingLLM())

    assert asyncio.run(uc.is_utterance_complete(conn, "空炮")) is True


def test_unparseable_verdict_fails_open():
    conn = _StubConn(llm=_FakeLLM("唔知呀"))

    assert asyncio.run(uc.is_utterance_complete(conn, "空炮")) is True


def test_missing_llm_fails_open():
    """No LLM wired (e.g. nointent config) must not block the turn."""
    conn = _StubConn(llm=None)

    assert asyncio.run(uc.is_utterance_complete(conn, "空炮")) is True


def test_empty_text_is_complete():
    """Empty/whitespace text must not be held forever."""
    conn = _StubConn(llm=_FakeLLM('{"complete": false}'))

    assert asyncio.run(uc.is_utterance_complete(conn, "")) is True
    assert asyncio.run(uc.is_utterance_complete(conn, "   ")) is True


def test_hanging_judge_times_out_and_fails_open():
    """⚠️ A HANG must fail open too, and must not block forever.

    The fail-open contract covers a stalled endpoint, not just an error: a
    judge that never returns would otherwise wedge the ASR turn indefinitely
    and the user would never be answered. `asyncio.wait_for` bounds it.
    """
    import time

    class _HangingLLM:
        async def response(self, messages):
            await asyncio.sleep(3600)  # never returns

    conn = _StubConn(llm=_HangingLLM())

    started = time.monotonic()
    verdict = asyncio.run(uc.is_utterance_complete(conn, "空炮"))
    elapsed = time.monotonic() - started

    assert verdict is True, "a hung judge must fail OPEN (answer the turn)"
    assert elapsed < uc._JUDGE_TIMEOUT_S + 2.0, (
        f"the judge was not bounded — took {elapsed:.1f}s "
        f"(timeout is {uc._JUDGE_TIMEOUT_S}s)"
    )
