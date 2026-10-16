"""Regression: the completeness judge must call the REAL LLM adapter interface.

## The bug (found 2026-10-16, present since the gate was written)

`judge_completeness_llm` called the adapter as:

    result = llm.response(messages)          # ONE argument

but every adapter in this repo subclasses `LLMProviderBase`, whose signature is:

    def response(self, session_id, dialogue)  # TWO required arguments

so the real call raised, on every single request:

    TypeError: response() missing 1 required positional argument: 'dialogue'

That `TypeError` was swallowed by the broad `except` in the judge and logged as
`completeness judge failed (...) — failing open` -> `None` -> treated as
COMPLETE. **Tier 2 therefore never once executed in production.** The gate was
reduced to tier 1 (the free trailing-token pre-filter), so any fragment that
looks syntactically whole — e.g. `我想睇下啲咩方案` — was answered immediately.

## Why the existing suite did not catch it

⚠️ `test_utterance_completeness.py` defines `_FakeLLM.response(self, messages)`
— a ONE-argument double written to mirror the buggy call. It cannot raise the
TypeError the real adapter raises, so all 12 of those tests pass on broken code.
A test double that mirrors the caller instead of the interface guards nothing.

These tests therefore drive a **real `LLMProviderBase` subclass**, so the
argument contract is enforced by the same base class production uses.
"""

import asyncio

import pytest

from core.providers.llm.base import LLMProviderBase
from core.utils import utterance_completeness as uc


class _FakeLogger:
    def __init__(self):
        self.warnings = []

    def bind(self, **kwargs):
        return self

    def info(self, msg):
        pass

    def warning(self, msg):
        self.warnings.append(msg)


class _StubConn:
    def __init__(self, llm=None):
        self.llm = llm
        self.logger = _FakeLogger()


class _RealInterfaceLLM(LLMProviderBase):
    """A genuine `LLMProviderBase` subclass — NOT a hand-rolled double.

    Inheriting means `response` keeps the base class's real signature
    `(session_id, dialogue)`. A caller passing the wrong shape raises the same
    TypeError production raises. Records what it was actually invoked with.
    """

    def __init__(self, verdict_json='{"complete": false}'):
        self.verdict_json = verdict_json
        self.received_session_id = None
        self.received_dialogue = None
        self.call_count = 0

    def response(self, session_id, dialogue):
        self.call_count += 1
        self.received_session_id = session_id
        self.received_dialogue = dialogue
        # Real adapters return a generator; the judge must consume it as one.
        yield self.verdict_json


# --------------------------------------------------------------------------
# The RED test for the TypeError
# --------------------------------------------------------------------------

def test_judge_calls_real_adapter_without_raising():
    """Driving the real interface must NOT raise.

    Before the fix this raised `TypeError: response() missing 1 required
    positional argument: 'dialogue'`, was swallowed, and silently resolved to
    COMPLETE. The judge's own failure log is the tell — assert on it so a
    regression to the one-arg call is caught even if a future refactor makes
    the TypeError non-fatal.
    """
    llm = _RealInterfaceLLM('{"complete": false}')
    conn = _StubConn(llm=llm)

    verdict = asyncio.run(uc.is_utterance_complete(conn, "空炮"))

    assert conn.logger.warnings == [], (
        f"the judge failed against the real adapter interface and failed open: "
        f"{conn.logger.warnings}"
    )
    assert llm.call_count == 1, "tier 2 must reach the adapter exactly once"
    assert verdict is False, (
        "the adapter said the utterance was INCOMPLETE but the gate answered it "
        "anyway — the judge's verdict is being discarded"
    )


def test_judge_passes_a_valid_dialogue_to_the_adapter():
    """The dialogue must be a well-formed chat-completions message list, not a
    bare string or a pre-shaped blob."""
    llm = _RealInterfaceLLM('{"complete": false}')
    conn = _StubConn(llm=llm)

    asyncio.run(uc.is_utterance_complete(conn, "空炮"))

    dialogue = llm.received_dialogue
    assert isinstance(dialogue, list) and dialogue, (
        f"dialogue must be a non-empty list, got {dialogue!r}"
    )
    roles = [m.get("role") for m in dialogue]
    assert roles == ["system", "user"], f"unexpected dialogue shape: {dialogue!r}"
    assert dialogue[-1]["content"] == "空炮", "the user's text must be forwarded"


def test_judge_consumes_a_generator_result():
    """⚠️ Adapters return GENERATORS. `str(generator)` is `<generator object>`,
    which `_parse_complete` would read as unparseable and silently fail open.

    This is the second, quieter half of the same bug: even with the argument
    shape fixed, a non-consumed generator verdict means the judge never really
    takes effect. The fix must join the stream.
    """
    llm = _RealInterfaceLLM('{"complete": false}')
    conn = _StubConn(llm=llm)

    verdict = asyncio.run(uc.is_utterance_complete(conn, "空炮"))

    assert verdict is False, (
        "the adapter yielded a parsable `false` verdict but the gate returned "
        "COMPLETE — the generator result is not being consumed/parsed"
    )


def test_judge_still_fails_open_when_adapter_explodes():
    """The fail-open contract must survive the fix — a genuinely broken adapter
    still degrades to COMPLETE rather than wedging the turn."""

    class _ExplodingRealInterface(LLMProviderBase):
        def response(self, session_id, dialogue):
            raise RuntimeError("endpoint down")
            yield  # pragma: no cover — makes this a generator

    conn = _StubConn(llm=_ExplodingRealInterface())
    logger = conn.logger

    assert asyncio.run(uc.is_utterance_complete(conn, "空炮")) is True
    assert logger.warnings, "an exploding adapter must still be logged"
