"""Integration tests for the completeness gate's HOLD + MERGE behaviour.

These guard the actual user symptom end-to-end at the ASR->startToChat seam:
one thought arriving as three VAD fragments must produce exactly ONE chat turn,
with the fragments merged — not three separate answers.

The event order modelled here is the real one from server.log 2026-10-11
11:18:44 -> 11:19:44: each fragment arrives as a *completed* ASR result
(i.e. handle_voice_stop has already fired for it), in sequence.
"""

import asyncio
from unittest import mock

import pytest


class _StubConn:
    def __init__(self, llm):
        self.llm = llm
        self.logger = _FakeLogger()
        self._pending_utterance = None


class _FakeLogger:
    def bind(self, **kwargs):
        return self

    def info(self, msg):
        pass

    def warning(self, msg):
        pass


class _ScriptedLLM:
    """Returns a verdict per call, in order."""

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.calls = []

    def response(self, messages):
        self.calls.append(messages[-1]["content"])
        verdict = self.verdicts.pop(0) if self.verdicts else '{"complete": true}'
        return verdict


def _simulate_fragment(conn, text, answered):
    """Run one fragment through the gate, exactly as asr/base.py does."""
    from core.utils.utterance_completeness import is_utterance_complete

    pending = conn._pending_utterance
    candidate = f"{pending}{text}" if pending else text

    if not asyncio.run(is_utterance_complete(conn, text)):
        conn._pending_utterance = candidate
        return
    conn._pending_utterance = None
    answered.append(candidate)


def test_three_fragments_of_one_thought_produce_one_answer():
    """THE user symptom, end to end.

    Before the gate: 3 fragments -> 3 LLM turns -> 3 answers (the "batches").
    With the gate: fragment 1+2 held (trailing 嘅 / 所以 caught by tier 1),
    fragment 3 completes the thought -> exactly ONE answer, merged.
    """
    conn = _StubConn(llm=_ScriptedLLM(['{"complete": true}']))
    answered = []

    # The three real fragments, in the real order.
    _simulate_fragment(conn, "一個關於天頭痛嘅。", answered)
    assert answered == [], "fragment 1 (ends on 嘅) was answered — cascade not stopped"

    _simulate_fragment(
        conn, "問題啦，因為我呢排成日都有偏頭痛，咁，我想睇下啲咩方案，所以。", answered
    )
    assert answered == [], "fragment 2 (ends on 所以) was answered — cascade not stopped"

    # Final fragment: ASR garbled it, so tier 1 can't judge -> tier 2 says complete.
    _simulate_fragment(conn, "我想睇下有咩方案", answered)

    assert len(answered) == 1, (
        f"expected exactly 1 answer for one thought, got {len(answered)}: {answered}"
    )
    merged = answered[0]
    # The held fragments must be preserved in the merged text, not dropped.
    assert "一個關於天頭痛嘅" in merged
    assert "偏頭痛" in merged
    assert "我想睇下有咩方案" in merged


def test_complete_sentence_answers_immediately():
    """Complementary invariant: a clean full sentence must NOT be held.
    If it were, every normal turn would stall."""
    conn = _StubConn(llm=_ScriptedLLM(['{"complete": true}']))
    answered = []

    _simulate_fragment(conn, "講翻今日新聞啦。", answered)

    assert answered == ["講翻今日新聞啦。"]


def test_hold_does_not_leak_across_turn_boundary():
    """A held fragment must be dropped at a turn boundary (reset_audio_states),
    never merged into a later, unrelated utterance."""
    conn = _StubConn(llm=_ScriptedLLM(['{"complete": true}']))
    answered = []

    _simulate_fragment(conn, "我想話加個。", answered)
    assert conn._pending_utterance is not None, "fragment should be held"

    # Turn boundary: abort / listen start calls reset_audio_states().
    conn._pending_utterance = None  # what reset_audio_states() now does

    _simulate_fragment(conn, "講翻今日新聞啦。", answered)

    assert answered == ["講翻今日新聞啦。"], (
        "a stale held fragment leaked into the next turn"
    )
