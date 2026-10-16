# Plan — Part B: ECAPA speaker gate before startToChat (v0.9.22)

**Repo:** `~/Projects/ai-where/server-upstream/main/xiaozhi-server`
**Rollback:** `git checkout v0.9.21` (cd2d1764, pushed)
**Risk:** MEDIUM-HIGH — a bad gate REJECTS Vincent (worse than no gate)
**Time:** ~90 min incl. RED→GREEN, build-tester, live verification

## Goal

Only Vincent's voice opens a turn. MTR/TV/other voices are rejected **before**
the turn is dispatched, so they cannot reach `startToChat` (and therefore cannot
abort an in-flight turn either — this subsumes the abandoned Part A).

## Why the existing voiceprint path does NOT do this

`asr/base.py:124-147` runs `voiceprint_provider.identify_speaker()` and then
attaches the result as a **label**:

```python
if speaker_name:
    raw_text["speaker"] = speaker_name     # line 153
...
await startToChat(conn, candidate)          # line 220 — fires REGARDLESS
```

The skill's rule: **a speaker-ID result used as a LABEL filters nothing.** The
turn is dispatched whether or not the voice matched. That is why MTR noise
became a turn tagged `未识别`.

## Design

New module `core/utils/ecapa_gate.py`, ported from the PROVEN
`~/Projects/butler-voice-profiles/ecapa_embed.py` (do not re-implement):

- `get_embedder()` — cached SpeechBrain EncoderClassifier (warm, thread-safe)
- `embed_wav_float(signal)` — 16k mono float32 -> 192-dim unit vector
- `cosine(a, b)`
- `load_speaker_dir(dir)` — {name: 192-dim}, dim-guarded (never mix 256-dim)

New `core/utils/speaker_gate.py` — the gate logic (pure, testable):

```python
def evaluate(emb, enrolled, threshold, margin) -> GateResult
```
- best score >= threshold AND (best - second) >= margin  -> admit
- else -> reject
- **fail-OPEN on no enrolment / embedder error** (never lock Vincent out of his
  own assistant if the model fails to load) — but LOG LOUDLY.

Wiring in `asr/base.py`:
- After `speaker_name` resolution, if `conn.speaker_gate_enabled`:
  compute embedding from `combined_pcm_data`, evaluate.
- **On reject: log + return BEFORE line 220** (`await startToChat`).

Config (`data/.config.yaml`):
```yaml
speaker_gate:
  enabled: true
  speaker_dir: ~/.cache/ai-where/speakers
  threshold: 0.42
  margin: 0.05
  min_audio_ms: 1000        # too-short audio cannot be judged -> fail open
```

## Threshold provenance (not invented)

- `0.42` — the documented ECAPA floor (skill + `speaker_match.py`).
- `0.05` — the documented ECAPA margin (`speaker_match.py:33`).
- Measured on the REAL enrolled print (2026-10-16):
  - held-out Vincent vs new print: **0.7776** (should ADMIT)
  - held-out Vincent vs old print: 0.3246 (would REJECT)
  So 0.42 sits far below the true-positive score — good headroom.

## ⚠️ Unverified assumption I must state

There is **no negative control**: only `vincent.npy` exists, so I have NOT
measured what a NON-Vincent voice scores against the new print. The skill
records user↔TV at −0.01…+0.22 and user↔other prints ≈0.1–0.25, which suggests
rejection will work — but that is a REASONED expectation, not a measurement on
this print. **A negative control is required before trusting rejection.**
Vincent to supply a short recording of another voice (Wayne, or a TV clip).

## Tests (RED first)

`tests/core/utils/test_speaker_gate.py`:
1. embed vector above threshold+margin -> ADMIT
2. below threshold -> REJECT
3. passes threshold but margin too small -> REJECT (near-tie)
4. no enrolled prints -> fail OPEN (with warning)
5. embedder raises -> fail OPEN (with error log), turn still answered
6. audio shorter than min_audio_ms -> fail OPEN (too short to judge)
7. dim guard: 256-dim file ignored, never compared against 192-dim

`tests/core/providers/asr/test_gate_blocks_turn.py`:
8. **rejected speaker -> `startToChat` NOT called** (the whole point)
9. admitted speaker -> `startToChat` IS called
10. gate disabled by config -> always calls through (backward compatible)

Each seen RED against pre-gate source.

## Explicitly NOT in scope
- Deleting the legacy label-only voiceprint path (leave working, separate task)
- Part A turn precedence — SUBSUMED: if noise is rejected it never reaches
  startToChat. The abandoned `turn_routing` attempt is NOT landed.
- Threshold auto-tuning

## Verification
- Full suite (expect 86 + new, 2 skipped)
- build-tester: independent RED verify + assess fail-open correctness
- Live: restart, confirm not locked out, watch a real turn admit
- **Negative control** with a second voice before declaring MTR fixed
