"""The ECAPA embedder must be WARM before the first turn arrives.

⚠️ WHY THIS FILE EXISTS (2026-10-16):
`get_embedder()` is lazy — the ECAPA model is not built until the first call to
`embed_pcm()`. That call happens on the TURN PATH, inside handle_voice_stop,
before `startToChat`. So the very first utterance after every server start pays
the full model-load cost inside the user's "thinking..." window.

Measured live on Vincent's device (2026-10-16 20:21):

    20:21:52  识别文本: I want to do my artulation training.
    20:21:53  声纹门控 ADMIT (...) | embed=1081ms    <- model load, on the turn
    20:21:53  大模型收到用户消息                      <- LLM starts only AFTER this
    20:21:57  识别文本: ... (next turn)
    20:21:57  声纹门控 ADMIT (...) | embed=46ms      <- warm

Vincent: "thinking.. is very long". Cold vs warm, measured directly:

    cold 1st call (model load): 1498ms
    warm 2nd call:                23ms     <- 65x faster

launchd KeepAlive restarts the server on every deploy, so the cold cost is paid
repeatedly during a testing session — once per session at minimum.

The fix: expose a `prewarm_embedder()` that drives the same lazy init during
startup, so the cost lands at boot (where nobody is waiting) rather than on the
first thing Vincent says.
"""

import os
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.utils import speaker_gate as SG  # noqa: E402


def test_prewarm_embedder_exists_and_returns_bool():
    """A prewarm entry point must exist — startup needs something to call."""
    assert hasattr(SG, "prewarm_embedder"), (
        "speaker_gate has no prewarm_embedder(); the ECAPA model can only be "
        "loaded lazily on the turn path, so the first turn after every restart "
        "pays the full model-load cost inside the user's 'thinking...' window"
    )
    assert callable(SG.prewarm_embedder)


def test_prewarm_actually_builds_the_embedder():
    """After prewarm, a call to embed_pcm must NOT pay model-load cost.

    This is the invariant that matters: prewarm must leave the embedder hot.
    """
    # force a cold state so the measurement is meaningful
    SG._embedder = None
    SG._embedder_failed = False

    ok = SG.prewarm_embedder()
    assert ok is True, "prewarm reported failure; the model did not build"
    assert SG.get_embedder() is not None, (
        "prewarm returned True but the embedder is still None"
    )


def test_warm_call_is_fast_after_prewarm():
    """A turn arriving after prewarm must be fast (< 250ms for 3s of audio).

    Guards the actual user-visible symptom: a warm embed is ~23ms, a cold one
    ~1500ms. If prewarm regresses, this catches it.
    """
    import numpy as np

    SG._embedder = None
    SG._embedder_failed = False
    assert SG.prewarm_embedder() is True

    pcm = (np.random.randn(16000 * 3) * 3000).astype(np.int16).tobytes()
    t0 = time.perf_counter()
    emb = SG.embed_pcm(pcm)
    dt_ms = (time.perf_counter() - t0) * 1000

    assert emb is not None, "embed_pcm returned None after a successful prewarm"
    assert dt_ms < 250, (
        f"first embed after prewarm took {dt_ms:.0f}ms — prewarm did not leave "
        f"the model hot (cold load is ~1500ms, warm ~23ms)"
    )


def test_prewarm_failure_is_not_fatal():
    """prewarm must return False, never raise — a missing model must not stop boot.

    Same fail-open principle as the gate itself: being unable to load the model
    degrades to 'gate disabled', it does not prevent the server from starting.
    """
    SG._embedder = None
    SG._embedder_failed = False

    with mock.patch.object(SG, "_build_embedder", side_effect=RuntimeError("boom")):
        result = SG.prewarm_embedder()  # must not raise

    assert result is False, "prewarm should report False when the model fails"
    # and it must not leave the gate permanently broken in a way that crashes
    SG._embedder = None
    SG._embedder_failed = False
