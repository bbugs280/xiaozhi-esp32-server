"""Integration test: the REAL handle_voice_stop must block a rejected voice.

⚠️ WHY THIS FILE EXISTS (build-tester finding, 2026-10-16):
The first version of these tests validated `evaluate()`'s arithmetic and a
LOCAL COPY of the caller logic. When the tester neutered the production
short-circuit in `asr/base.py` —

    if not _verdict.admitted:
        return          # <- replaced with `pass`

— the non-Vincent turn LEAKED through, and the committed suite still reported
97 passed. A test that never ran RED against the buggy code guards nothing
(AGENTS.md Iron Law). The most important line of the change was guarded only
by an untracked /tmp scratch script.

This file drives the production path directly, so deleting the short-circuit
makes THESE tests fail.
"""

import asyncio
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.providers.asr import base as B  # noqa: E402

SPEAKER_DIR = os.path.expanduser("~/.cache/ai-where/speakers")


def _unit(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=192).astype(np.float32)
    return v / np.linalg.norm(v)


def _pcm(seconds=3.0, sr=16000):
    """Synthetic PCM — content is irrelevant because we stub the embedder."""
    n = int(seconds * sr)
    return (np.zeros(n, dtype=np.int16)).tobytes()


class _Conn:
    """Minimal ConnectionHandler for handle_voice_stop."""

    def __init__(self, enabled=True):
        self.speaker_gate_enabled = enabled
        self.speaker_gate_speaker_dir = SPEAKER_DIR
        self.speaker_gate_threshold = 0.42
        self.speaker_gate_margin = 0.05
        self.speaker_gate_min_audio_ms = 1000
        self.sample_rate = 16000
        self.session_id = "test-session"
        self.voiceprint_provider = None
        self.client_listen_mode = "auto"
        self.read_config_from_api = False
        self.client_have_voice = True
        self.client_voice_stop = True

    def reset_audio_states(self):
        pass


class _FakeASR(B.ASRProviderBase):
    async def speech_to_text(self, *a, **k):
        return ({"content": "x"}, None)

    def stop_ws_connection(self):
        pass


def _provider():
    prov = _FakeASR.__new__(_FakeASR)
    prov.speech_to_text_wrapper = mock.AsyncMock(
        return_value=({"content": "Exation central interchange station", "language": "en"}, None)
    )
    prov._pcm_to_wav = lambda b: b"RIFF"
    prov._wrap_content = lambda t, s: t
    prov._unwrap_content = lambda t: t
    return prov


def _run(conn, embed_value):
    """Drive the real handle_voice_stop with a stubbed embedding."""
    dispatched = []

    async def fake_start(c, text):
        dispatched.append(text)

    with mock.patch.object(B, "startToChat", fake_start), \
         mock.patch("core.utils.speaker_gate.embed_pcm", return_value=embed_value):
        asyncio.run(_provider().handle_voice_stop(conn, [_pcm()]))

    return dispatched


def test_rejected_voice_never_reaches_start_to_chat():
    """THE INVARIANT: a voice below threshold must NOT be dispatched.

    Neuter the `if not _verdict.admitted: return` short-circuit in
    core/providers/asr/base.py and this test FAILS — that is the point.
    """
    conn = _Conn(enabled=True)
    # embedding far from every enrolled print -> rejected
    dispatched = _run(conn, _unit(999))

    assert dispatched == [], (
        "a rejected voice reached startToChat — the production short-circuit "
        "in asr/base.py is not working (or was removed). This is the entire "
        "purpose of the gate."
    )


def test_admitted_voice_reaches_start_to_chat():
    """Positive control: the enrolled speaker must still be answered."""
    from core.utils import speaker_gate as SG  # noqa: PLC0415

    enrolled = SG.load_voiceprints(SPEAKER_DIR)
    if "vincent" not in enrolled:
        pytest.skip("no enrolled vincent print available")

    conn = _Conn(enabled=True)
    dispatched = _run(conn, enrolled["vincent"])

    assert dispatched, (
        "the enrolled speaker was blocked — the gate must admit Vincent, "
        "otherwise he is locked out of his own assistant"
    )


def test_gate_disabled_never_blocks():
    """With the gate off, behaviour must be exactly as before the feature."""
    conn = _Conn(enabled=False)
    dispatched = _run(conn, _unit(999))

    assert dispatched, "gate disabled must never block a turn"


def test_gate_disabled_does_not_even_compute_an_embedding():
    """Cost check: a disabled gate must not pay for ECAPA inference."""
    conn = _Conn(enabled=False)
    embed = mock.MagicMock(return_value=_unit(1))

    async def fake_start(c, text):
        pass

    with mock.patch.object(B, "startToChat", fake_start), \
         mock.patch("core.utils.speaker_gate.embed_pcm", embed):
        asyncio.run(_provider().handle_voice_stop(conn, [_pcm()]))

    assert embed.call_count == 0, (
        "a disabled gate still ran the embedder — wasted inference on every turn"
    )


def test_short_audio_fails_open_in_production_path():
    """Too-short audio cannot be judged; the turn must proceed."""
    conn = _Conn(enabled=True)
    dispatched = []

    async def fake_start(c, text):
        dispatched.append(text)

    with mock.patch.object(B, "startToChat", fake_start), \
         mock.patch("core.utils.speaker_gate.embed_pcm", return_value=_unit(999)):
        asyncio.run(_provider().handle_voice_stop(conn, [_pcm(seconds=0.2)]))

    assert dispatched, (
        "audio shorter than min_audio_ms must fail OPEN — dropping a legitimate "
        "short command is worse than admitting noise"
    )
