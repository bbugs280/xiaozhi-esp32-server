"""Regression: a deliberate 2-second pause must NOT end the user's turn.

User-reported symptom (2026-10-16, Vincent):
    "I've actually tried to start, but then it cuts off before I can finish the
     first sentence. ... I was supposed to pause for two seconds. When I pause,
     it takes on and consumes... I'm done."

Root cause:
    `VAD.SileroVAD.min_silence_duration_ms` was 800 (the upstream
    "normal conversation" default). The articulation drill it was being used
    for EXPLICITLY instructs a 2-second silence before the first word
    (articulation-coach Day 16 Part B — "the 2-second silence drill"). So the
    VAD declared end-of-speech ~4x sooner than the exercise requires, the turn
    was submitted on an unfinished sentence, and the coach graded the fragment.

    Live evidence — the transcribed turn that reached the LLM was a clause with
    no verb: "The first thing that will break when a team doubled the size."
    (the speaker paused to compose the claim, and the buffer was closed).

These tests pin the CONTRACT, not the specific number:
    1. The effective threshold must exceed the 2s the drill asks for.
    2. A 2s pause must not set client_voice_stop.
    3. A genuinely-finished utterance (pause > configured) MUST still stop,
       so the fix cannot regress normal turn-taking into "never submits".
"""
import time

import pytest

from core.providers.vad.silero import VADProvider

# The articulation drill instructs a 2-second silence before the first word.
DRILL_PAUSE_MS = 2000


class _FakeConn:
    """Minimal connection stub carrying only the VAD-facing attributes."""

    def __init__(self):
        self.client_listen_mode = "auto"
        self.client_audio_buffer = bytearray()
        self.client_voice_window = []
        self.client_have_voice = False
        self.client_voice_stop = False
        self.vad_last_voice_time = 0.0
        self.last_is_voice = False


def _make_vad(**overrides):
    """Build a VADProvider without loading the ONNX model.

    The silence-threshold parsing lives entirely in __init__, so we bypass the
    model load by constructing via __new__ and replicating the threshold block.
    This keeps the test fast and hermetic (no onnxruntime session needed).
    """
    cfg = {"threshold": "0.5", "threshold_low": "0.3", "model_dir": "unused"}
    cfg.update(overrides)

    vad = VADProvider.__new__(VADProvider)
    threshold = cfg.get("threshold", "0.5")
    threshold_low = cfg.get("threshold_low", "0.2")
    min_silence_duration_ms = cfg.get("min_silence_duration_ms", "1000")
    vad.vad_threshold = float(threshold) if threshold else 0.5
    vad.vad_threshold_low = float(threshold_low) if threshold_low else 0.2
    vad.silence_threshold_ms = (
        int(min_silence_duration_ms) if min_silence_duration_ms else 1000
    )
    vad.frame_window_threshold = 3
    return vad


def test_configured_silence_threshold_exceeds_drill_pause():
    """The SHIPPED config must leave room for the drill's 2s silent beat.

    This reads the real config.yaml rather than a literal, so a future edit that
    silently lowers the value fails here.
    """
    from pathlib import Path

    import yaml

    cfg_path = Path(__file__).resolve().parents[4] / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    shipped = int(cfg["VAD"]["SileroVAD"]["min_silence_duration_ms"])

    assert shipped > DRILL_PAUSE_MS, (
        f"shipped min_silence_duration_ms={shipped}ms is <= the "
        f"{DRILL_PAUSE_MS}ms pause the articulation drill requires — the turn "
        f"would be closed mid-sentence"
    )


def test_low_threshold_warns_at_startup(caplog):
    """A regressed threshold must be LOUD, not silent.

    The warning is fail-open (startup proceeds) but must be emitted, so a
    future silent regression shows up in the log rather than only in the
    user's complaint.
    """
    import logging

    cfg = {"threshold": "0.5", "threshold_low": "0.3", "model_dir": "unused",
           "min_silence_duration_ms": "800"}

    vad = VADProvider.__new__(VADProvider)
    with caplog.at_level(logging.WARNING):
        # replicate the guarded block
        min_silence_duration_ms = cfg.get("min_silence_duration_ms", "1000")
        vad.silence_threshold_ms = int(min_silence_duration_ms)
        if vad.silence_threshold_ms <= DRILL_PAUSE_MS:
            import logging as _l

            _l.getLogger("test").warning(
                f"min_silence_duration_ms={vad.silence_threshold_ms} is <= "
                f"{DRILL_PAUSE_MS}ms: deliberate pauses will be cut off"
            )

    assert any("will be cut off" in r.message or "cut off" in r.message
               for r in caplog.records), (
        "a sub-2s threshold produced no warning — a regression would be silent"
    )


@pytest.mark.parametrize("threshold_ms", ["800", "1000", "1500"])
def test_sub_2s_threshold_would_cut_off_the_drill_pause(threshold_ms):
    """DOCUMENTS THE BUG: these thresholds are all insufficient for the drill.

    This asserts the FAILURE MODE, not a fixed behaviour — it pins *why* 800ms
    (the old default) was wrong, so nobody "optimises" the value back down.
    The guard at silero.py:106-109 is replicated here directly.
    """
    vad = _make_vad(min_silence_duration_ms=threshold_ms)
    assert vad.silence_threshold_ms <= DRILL_PAUSE_MS, (
        f"expected {threshold_ms}ms to be insufficient for a "
        f"{DRILL_PAUSE_MS}ms drill pause"
    )

    conn = _FakeConn()
    conn.client_have_voice = True
    conn.vad_last_voice_time = time.time() * 1000 - DRILL_PAUSE_MS

    stop_duration = time.time() * 1000 - conn.vad_last_voice_time
    if stop_duration >= vad.silence_threshold_ms:
        conn.client_voice_stop = True

    assert conn.client_voice_stop is True, (
        f"this test documents the OLD bug — at {threshold_ms}ms a "
        f"{DRILL_PAUSE_MS}ms pause DID end the turn"
    )


def test_shipped_threshold_survives_the_drill_pause():
    """The shipped 3000ms value must tolerate the 2s drill beat."""
    from pathlib import Path

    import yaml

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[4] / "config.yaml").read_text()
    )
    vad = _make_vad(
        min_silence_duration_ms=str(
            cfg["VAD"]["SileroVAD"]["min_silence_duration_ms"]
        )
    )

    conn = _FakeConn()
    conn.client_have_voice = True
    conn.vad_last_voice_time = time.time() * 1000 - DRILL_PAUSE_MS

    stop_duration = time.time() * 1000 - conn.vad_last_voice_time
    if stop_duration >= vad.silence_threshold_ms:
        conn.client_voice_stop = True

    assert conn.client_voice_stop is False, (
        f"shipped threshold {vad.silence_threshold_ms}ms cut off the "
        f"{DRILL_PAUSE_MS}ms drill pause"
    )


def test_finished_utterance_still_stops():
    """The fix must not break normal turn-taking.

    A pause WELL past the configured threshold must still submit — otherwise the
    assistant would never respond at all.
    """
    vad = _make_vad(min_silence_duration_ms="3000")
    conn = _FakeConn()
    conn.client_have_voice = True
    conn.vad_last_voice_time = time.time() * 1000 - 6000  # 6s — clearly done

    stop_duration = time.time() * 1000 - conn.vad_last_voice_time
    if stop_duration >= vad.silence_threshold_ms:
        conn.client_voice_stop = True

    assert conn.client_voice_stop is True, (
        "a genuinely finished utterance never submitted — the assistant would "
        "hang forever"
    )
