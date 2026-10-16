"""RED: the speaker gate must ADMIT Vincent and REJECT everyone else.

The whole point of this feature is REJECTION. Vincent's stated driver:
"only my voice … rather than having my radio, my TV, or people talking to me
get registered."

The server ALREADY had a voiceprint path, but it only attached a LABEL:

    asr/base.py:153   raw_text["speaker"] = speaker_name
    asr/base.py:220   await startToChat(conn, candidate)   <- fires regardless

So an MTR announcement became a turn tagged "未识别" and was answered anyway.
A label filters nothing. The gate must SHORT-CIRCUIT before the turn is
dispatched.

Measured on the real enrolled print (2026-10-16):
  held-out Vincent vs new print  : 0.7776  -> must ADMIT
  held-out Vincent vs old print  : 0.3246  -> would REJECT
Threshold 0.42 / margin 0.05 are the documented ECAPA values, not invented.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _unit(seed):
    """Deterministic 192-dim unit vector."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=192).astype(np.float32)
    return v / np.linalg.norm(v)


def _gate():
    from core.utils import speaker_gate  # noqa: PLC0415

    return speaker_gate


# --------------------------------------------------------------------------
# evaluate(): the pure decision function
# --------------------------------------------------------------------------


def test_admits_when_above_threshold_and_margin():
    """Vincent's real held-out score (0.7776) must be admitted."""
    g = _gate()
    vincent = _unit(1)
    # a different speaker, clearly separated (real cross-sims are ~0.1-0.25)
    other = _unit(2)
    enrolled = {"vincent": vincent, "wayne": other}

    res = g.evaluate(vincent, enrolled, threshold=0.42, margin=0.05)

    assert res.admitted is True
    assert res.speaker == "vincent"
    assert res.score > 0.9


def test_rejects_when_below_threshold():
    """A voice scoring under the 0.42 floor (like the OLD print at 0.3246)
    must be rejected — it must not open a turn."""
    g = _gate()
    vincent = _unit(1)
    # query is orthogonal-ish to the only enrolled print
    query = _unit(99)
    enrolled = {"vincent": vincent}

    res = g.evaluate(query, enrolled, threshold=0.42, margin=0.05)

    assert res.admitted is False, "below-threshold voice must NOT open a turn"
    assert res.speaker is None


def test_rejects_near_tie_despite_passing_threshold():
    """Margin rule: if two prints score close, do not guess. Vincent's own
    skill documents father/son overlap (0.729 ECAPA) as a real risk."""
    g = _gate()
    a = _unit(1)
    # craft a second print that scores just under a, both above threshold
    b = a * 0.8 + _unit(5) * 0.2
    b = (b / np.linalg.norm(b)).astype(np.float32)
    enrolled = {"vincent": a, "wayne": b}

    res = g.evaluate(a, enrolled, threshold=0.42, margin=0.05)

    assert res.admitted is False, (
        "a near-tie must be rejected — the margin rule exists to avoid "
        "misrouting when two voices are too similar"
    )


def test_fails_open_when_no_enrolment():
    """No prints on disk must NEVER lock Vincent out of his own assistant."""
    g = _gate()

    res = g.evaluate(_unit(1), {}, threshold=0.42, margin=0.05)

    assert res.admitted is True, (
        "with no enrolled prints the gate must FAIL OPEN — locking the user "
        "out of their own assistant is worse than admitting noise"
    )
    assert res.reason is not None and "no" in res.reason.lower()


def test_fails_open_when_audio_too_short():
    """Too-short audio cannot be judged reliably (ECAPA needs ~3s+). Rejecting
    on it would drop legitimate short commands."""
    g = _gate()
    enrolled = {"vincent": _unit(1)}

    res = g.evaluate(
        _unit(99), enrolled, threshold=0.42, margin=0.05,
        audio_ms=300, min_audio_ms=1000,
    )

    assert res.admitted is True, "audio below min_audio_ms must fail open"
    assert "short" in res.reason.lower()


def test_rejects_when_audio_is_long_enough_and_voice_wrong():
    """The complement of the previous test: long enough audio + wrong voice
    must still be rejected (the fail-open must not swallow real rejections)."""
    g = _gate()
    enrolled = {"vincent": _unit(1)}

    res = g.evaluate(
        _unit(99), enrolled, threshold=0.42, margin=0.05,
        audio_ms=4000, min_audio_ms=1000,
    )

    assert res.admitted is False


def test_load_speaker_dir_ignores_wrong_dimension(tmp_path):
    """Dim guard: a legacy resemblyzer 256-dim print must NEVER be compared
    against a 192-dim ECAPA embedding (spaces are incompatible)."""
    g = _gate()
    np.save(tmp_path / "vincent.npy", _unit(1))
    np.save(tmp_path / "legacy.npy", np.random.default_rng(7).normal(size=256).astype(np.float32))

    loaded = g.load_voiceprints(str(tmp_path))

    assert "vincent" in loaded
    assert "legacy" not in loaded, (
        "a 256-dim resemblyzer print must be skipped, not compared against "
        "192-dim ECAPA embeddings"
    )
    assert all(v.shape == (192,) for v in loaded.values())


def test_evaluate_result_is_serialisable_for_logging():
    """The decision must carry enough to log WHY, for live debugging."""
    g = _gate()
    res = g.evaluate(_unit(1), {"vincent": _unit(1)}, threshold=0.42, margin=0.05)

    assert hasattr(res, "score")
    assert hasattr(res, "reason")
    assert isinstance(res.reason, str)
    assert res.reason != ""
