"""ECAPA speaker gate — decide whether a voice may OPEN a turn.

Ported from the PROVEN implementation in ~/Projects/butler-voice-profiles
(`ecapa_embed.py`, `speaker_match.py`). Do not re-implement the embedder — the
Butler version is the working, live-tested one. This module adds the thing the
server was missing: a DECISION that blocks the turn.

⚠️ Why this exists (2026-10-16): the server already ran a voiceprint check but
used the result only as a LABEL:

    asr/base.py   raw_text["speaker"] = speaker_name
    asr/base.py   await startToChat(conn, candidate)   # fires regardless

So a subway announcement became a turn tagged "未识别" and was answered anyway.
Vincent: "it picked up background noise in MTR". A label filters nothing.

Design rules:
  - FAIL OPEN. If the model cannot load, no prints exist, or the audio is too
    short to judge, ADMIT the turn. Locking Vincent out of his own assistant is
    a worse failure than occasionally admitting noise. Every fail-open is logged.
  - Dim-guard the prints. resemblyzer (256-dim) and ECAPA (192-dim) embedding
    spaces are INCOMPATIBLE; mixing them silently produces garbage scores.
  - Threshold is a FLOOR, not a discriminator. ECAPA gives the right speaker
    ~0.40-0.90 and wrong speakers ~0.1-0.25, so the MARGIN rule (best - second)
    is what actually separates near-ties (father/son measure ~0.73 in ECAPA).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

TAG = __name__

SAMPLE_RATE = 16000
EMBED_DIM = 192

# Documented ECAPA values (see the butler-speaker-id-triage skill and
# speaker_match.py). NOT invented here.
DEFAULT_THRESHOLD = 0.42
DEFAULT_MARGIN = 0.05


@dataclass
class GateResult:
    """The decision, plus enough context to log WHY it was made."""

    admitted: bool
    speaker: Optional[str]
    score: float
    margin: float
    reason: str


# --------------------------------------------------------------------------
# embedder (lazy, cached, thread-safe)
# --------------------------------------------------------------------------

_embedder_lock = threading.Lock()
_embedder = None
_embedder_failed = False


def _build_embedder():
    from speechbrain.inference.speaker import EncoderClassifier  # noqa: PLC0415

    import torch  # noqa: PLC0415

    torch.set_num_threads(int(os.environ.get("ECAPA_THREADS", "1")))
    model_dir = os.environ.get("ECAPA_MODEL_DIR", "/tmp/ecapa_model")
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=model_dir,
        run_opts={"device": "cpu"},
    )


def get_embedder():
    """Cached warm ECAPA classifier. Returns None if it cannot be built."""
    global _embedder, _embedder_failed
    if _embedder_failed:
        return None
    with _embedder_lock:
        if _embedder is None:
            try:
                _embedder = _build_embedder()
            except Exception:
                _embedder_failed = True
                return None
        return _embedder


def prewarm_embedder() -> bool:
    """Drive the lazy embedder init at STARTUP, off the turn path.

    ⚠️ WHY (2026-10-16): get_embedder() is lazy, so the ECAPA model was built
    on the FIRST CALL to embed_pcm() — which happens inside handle_voice_stop,
    BEFORE startToChat. The first utterance after every server start therefore
    paid the full model-load cost inside Vincent's "thinking..." window.

    Measured on his device (2026-10-16 20:21):

        embed=1081ms   <- first turn after restart (model load, on the turn)
        embed=46ms     <- every turn after

    and directly on the host:

        cold 1st call (model load): 1498ms
        warm 2nd call:                23ms     <- 65x faster

    launchd KeepAlive restarts the server on every deploy, so this was paid
    repeatedly across a testing session. Prewarming moves the cost to boot,
    where nobody is waiting for a reply.

    Returns True if the model is warm, False if it could not be built. NEVER
    raises: a missing model must degrade to "gate disabled", not stop the
    server from booting (same fail-open principle as the gate itself).
    """
    try:
        return get_embedder() is not None
    except Exception:
        # NOTE: catches Exception, not BaseException — SystemExit /
        # KeyboardInterrupt still propagate. That matches get_embedder()'s
        # own handler, and this runs on a daemon thread, so a stray
        # BaseException cannot escape boot. Kept consistent deliberately.
        return False


def embed_pcm(pcm_bytes: bytes) -> Optional[np.ndarray]:
    """int16 little-endian 16k mono PCM -> 192-dim unit vector, or None."""
    if not pcm_bytes:
        return None
    cls = get_embedder()
    if cls is None:
        return None
    try:
        import torch  # noqa: PLC0415

        sig = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if sig.size == 0:
            return None
        tensor = torch.from_numpy(sig).unsqueeze(0)
        emb = cls.encode_batch(tensor).squeeze().cpu().numpy().astype(np.float32)
        n = np.linalg.norm(emb)
        if not np.isfinite(n) or n <= 0:
            return None
        return emb / n
    except Exception:
        return None


def load_voiceprints(speaker_dir: str, expected_dim: int = EMBED_DIM) -> Dict[str, np.ndarray]:
    """Load all *.npy as {name: 192-dim}; SKIP wrong-dim files.

    The dim guard is load-bearing: a legacy 256-dim resemblyzer print compared
    against a 192-dim ECAPA embedding yields a meaningless cosine, which would
    then be compared to a threshold and produce nonsense decisions.
    """
    out: Dict[str, np.ndarray] = {}
    if not speaker_dir or not os.path.isdir(speaker_dir):
        return out
    for f in sorted(os.listdir(speaker_dir)):
        if not f.endswith(".npy"):
            continue
        try:
            arr = np.load(os.path.join(speaker_dir, f))
        except Exception:
            continue
        if arr.shape == (expected_dim,):
            v = arr.astype(np.float32)
            n = np.linalg.norm(v)
            if np.isfinite(n) and n > 0:
                out[f[:-4]] = v / n
    return out


def evaluate(
    embedding: Optional[np.ndarray],
    enrolled: Dict[str, np.ndarray],
    threshold: float = DEFAULT_THRESHOLD,
    margin: float = DEFAULT_MARGIN,
    audio_ms: Optional[int] = None,
    min_audio_ms: int = 1000,
) -> GateResult:
    """Decide whether `embedding` may open a turn.

    Fails OPEN (admit) whenever a verdict cannot be trusted, and says why.
    """
    if embedding is None:
        return GateResult(True, None, 0.0, 0.0,
                          "no embedding (model unavailable) — failing open")

    if audio_ms is not None and audio_ms < min_audio_ms:
        return GateResult(True, None, 0.0, 0.0,
                          f"audio too short to judge ({audio_ms}ms < {min_audio_ms}ms) — failing open")

    if not enrolled:
        return GateResult(True, None, 0.0, 0.0,
                          "no enrolled voiceprints — failing open")

    e = np.asarray(embedding, dtype=np.float32).flatten()
    scores = []
    for name, ref in enrolled.items():
        r = np.asarray(ref, dtype=np.float32).flatten()
        if r.shape != e.shape:
            continue
        scores.append((name, float(np.dot(e, r))))

    if not scores:
        return GateResult(True, None, 0.0, 0.0,
                          "no dimension-compatible voiceprints — failing open")

    scores.sort(key=lambda t: -t[1])
    best_name, best = scores[0]
    second = scores[1][1] if len(scores) > 1 else 0.0
    m = best - second

    if best < threshold:
        return GateResult(False, None, best, m,
                          f"below threshold ({best:.4f} < {threshold}) — rejecting")
    # ⚠️ The margin rule guards near-ties (ECAPA father/son overlap measures
    # ~0.73). It is INERT while only ONE print is enrolled: second == 0.0, so
    # m == best, which has already passed the threshold check above. It starts
    # doing real work as soon as a 2nd speaker is enrolled. Kept because the
    # failure mode it prevents (silently routing to the wrong person) is worse
    # than the cost of the check.
    if m < margin:
        return GateResult(False, None, best, m,
                          f"margin too small ({m:.4f} < {margin}) — rejecting near-tie")

    return GateResult(True, best_name, best, m,
                      f"admitted as {best_name} ({best:.4f}, margin {m:.4f})")
