"""The gate must LOG admitted turns at INFO, with the score and embed cost.

⚠️ WHY THIS FILE EXISTS (2026-10-16):
The gate only logged REJECTIONS at INFO. The admit path logged at DEBUG
(`logger.bind(tag=TAG).debug(...)`), and production runs at INFO — so every
accepted turn was INVISIBLE in the log.

Why that mattered: the only in-production evidence was the reject cluster
(0.04-0.11). Nobody ever saw a single admitted score from real speech, so
"should we raise threshold 0.42?" was unanswerable from data — any change
would have been a GUESS, and a wrong guess locks Vincent out of his own
assistant.

The fix: log ADMIT at INFO with the score, the threshold it beat, and the
embed duration. REJECT keeps the same shape so both are comparable at a glance.

This drives the REAL handle_voice_stop (not a copy of the caller logic —
build-tester taught us that lesson) and captures LOGURU (the project's logger,
via config.logger.setup_logging), not stdlib logging.
"""

import asyncio
import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from loguru import logger as _loguru

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
    n = int(seconds * sr)
    return (np.zeros(n, dtype=np.int16)).tobytes()


class _Conn:
    def __init__(self):
        self.speaker_gate_enabled = True
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
        return_value=({"content": "hello there", "language": "en"}, None)
    )
    prov._pcm_to_wav = lambda b: b"RIFF"
    prov._wrap_content = lambda t, s: t
    prov._unwrap_content = lambda t: t
    return prov


def _gate_lines(embed_value, level="INFO"):
    """Run the real path; return gate log lines at >= `level` from LOGURU."""
    conn = _Conn()
    captured = []
    sink_id = _loguru.add(
        lambda m: captured.append((m.record["level"].name, m.record["message"])),
        level=level,
    )
    try:

        async def fake_start(c, text):
            pass

        with mock.patch.object(B, "startToChat", fake_start), \
             mock.patch(
                 "core.utils.speaker_gate.embed_pcm", return_value=embed_value
             ):
            asyncio.run(_provider().handle_voice_stop(conn, [_pcm()]))
    finally:
        _loguru.remove(sink_id)

    return [msg for lvl, msg in captured if "声纹门控" in msg]


def _vincent_print():
    from core.utils import speaker_gate as SG

    enrolled = SG.load_voiceprints(SPEAKER_DIR)
    if "vincent" not in enrolled:
        return None
    return np.asarray(enrolled["vincent"], dtype=np.float32)


def test_admitted_turn_is_logged_at_info_with_score():
    """THE INVARIANT: an admitted turn must leave an INFO trace with its score.

    Revert the admit log to `.debug(...)` and this test FAILS — which is the
    exact regression that made threshold tuning impossible.
    """
    vincent = _vincent_print()
    if vincent is None:
        pytest.skip("no enrolled print to test admission against")

    lines = _gate_lines(vincent)

    assert lines, (
        "an ADMITTED turn produced no INFO log line — the admit path is still "
        "at debug level or removed, so production shows nothing for accepted "
        "speech and the threshold cannot be tuned from data"
    )
    joined = " | ".join(lines)
    assert any("ADMIT" in ln for ln in lines), (
        f"no admit line found among: {lines!r}"
    )
    # must carry an actual numeric score (e.g. 0.8213, or 1.0000 for a perfect
    # self-match), not just a bare verdict word.
    import re as _re

    assert _re.search(r"\d+\.\d+", joined), (
        f"admit line carries no numeric score: {joined!r}"
    )


def test_rejected_turn_still_logged_at_info():
    """Regression guard: the reject path must keep working."""
    lines = _gate_lines(_unit(999))
    assert any("拒绝" in ln or "REJECT" in ln for ln in lines), (
        f"the REJECT path stopped logging at INFO: {lines!r}"
    )


def test_admit_and_reject_lines_share_shape():
    """Both verdicts must be greppable by one prefix, for log-based tuning."""
    vincent = _vincent_print()
    if vincent is None:
        pytest.skip("no enrolled print")

    admit = _gate_lines(vincent)
    reject = _gate_lines(_unit(999))

    for name, lines in (("admit", admit), ("reject", reject)):
        assert lines, f"no line for {name}"
        assert any("声纹门控" in ln for ln in lines), (
            f"{name} line lost the shared 声纹门控 prefix: {lines!r}"
        )
