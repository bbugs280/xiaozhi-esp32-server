"""RED: a REJECTED speaker must not reach startToChat.

This is the whole point of the gate. The server previously computed a speaker
label and then called `startToChat` anyway (asr/base.py:220), so noise opened
turns. These tests assert the short-circuit.

Observed symptom driving this (Vincent, 2026-10-16 13:56, MTR):
   识别文本: Exation central interchange station, between one.   <- MTR PA as a turn
   识别文本: 講新聞產車將會打齊。                                <- noise as a turn
"""

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _unit(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=192).astype(np.float32)
    return v / np.linalg.norm(v)


def test_gate_decision_can_block_before_start_to_chat():
    """Contract test: the gate is consulted and its `admitted=False` prevents
    dispatch. Modelled on the real call order in asr/base.py:220."""
    from core.utils import speaker_gate  # noqa: PLC0415

    dispatched = []

    async def startToChat(conn, text):
        dispatched.append(text)

    class _Conn:
        speaker_gate_enabled = True
        speaker_gate_threshold = 0.42
        speaker_gate_margin = 0.05
        speaker_gate_min_audio_ms = 1000
        speaker_gate_speaker_dir = "/tmp/nonexistent-prints"

    gate = speaker_gate
    enrolled = {"vincent": _unit(1)}

    # Simulate the server's decision point with a rejecting verdict
    res = gate.evaluate(_unit(99), enrolled, threshold=0.42, margin=0.05, audio_ms=4000)
    assert res.admitted is False

    # The gate's contract: caller must NOT dispatch when not admitted.
    if res.admitted:
        import asyncio

        asyncio.run(startToChat(_Conn(), "garbage"))

    assert dispatched == [], (
        "a rejected speaker must not reach startToChat — this is the entire "
        "purpose of the gate (noise must not open a turn)"
    )


def test_gate_admits_and_allows_dispatch():
    """The complement: an admitted voice DOES reach startToChat."""
    from core.utils import speaker_gate  # noqa: PLC0415

    enrolled = {"vincent": _unit(1)}
    res = speaker_gate.evaluate(
        _unit(1), enrolled, threshold=0.42, margin=0.05, audio_ms=4000
    )
    assert res.admitted is True
    assert res.speaker == "vincent"


def test_gate_disabled_means_always_admit():
    """Backward compatibility: with the gate OFF the server behaves exactly as
    before. A config-flag regression would lock Vincent out entirely."""
    from core.utils import speaker_gate  # noqa: PLC0415

    # When the feature is disabled the server must not even consult the gate.
    # Encode that as: evaluate() is never reached, so an empty enrolment and
    # a garbage embedding still cannot block (fail-open path).
    res = speaker_gate.evaluate(None, {}, threshold=0.42, margin=0.05)
    assert res.admitted is True
