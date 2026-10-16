"""RED tests for the TTS retry loop (2026-10-16).

## The production symptom

Vincent, 2026-10-16 12:03:36-12:04:00 — a **24-second silence** in one "thinking"
round, then:

    12:04:00  语音生成失败1次: 今日天氣唔錯。，错误: Edge TTS请求失败: 403
    12:04:00  语音生成失败2次: ... 403
    12:04:00  语音生成失败3次: ... 403
    12:04:00  语音生成失败4次: ... 403
    12:04:00  语音生成失败5次: ... 403
    12:04:00  ERROR 语音生成失败: 今日天氣唔錯。，请检查网络或服务是否正常

Note all five land in the SAME SECOND.

## Defect 1 — the retry loop cannot succeed and costs time

`core/providers/tts/base.py` retries **5 times back-to-back with no delay**
(`while max_repeat_time > 0`, no `await asyncio.sleep`). Against an upstream
that just returned `403 Invalid response status`, an immediate identical retry
is guaranteed to fail identically. The loop therefore has a ~0% success rate
for the failure mode it exists for, while always costing the user the full
retry budget before the client is told anything.

This is the SAME class of defect as the completeness judge removed in v0.9.19:
a mechanism whose failure mode is pure added delay. It should either retry
with backoff (a genuinely different attempt) or fail fast.

## Defect 2 — the `tts error` signal is missing on one of two exhaustion paths

`_emit_tts_error` is called at ONE site only (the `delete_audio_file=True`
branch). The **file branch** (`else:`, ~line 188) reports the failure to
`server.log` but never tells the client:

    logger.bind(tag=TAG).error(f"语音生成失败: {original_text}，...")
    # no self._emit_tts_error(...)

A client that gets no terminal signal waits forever — the EXACT original bug
the v0.9.16 `tts error` signal was added to fix, still open on the other branch.

⚠️ Production currently uses `delete_audio_file=True` (`connection.py:791`), so
Vincent hits the covered path. But the uncovered branch is a live code path for
other TTS providers and would reintroduce the spinner-forever bug.

⚠️ These tests are RED against the current code.
"""

import asyncio
import inspect
import time
from unittest import mock

import pytest

from core.providers.tts import base as tts_base


class _Logger:
    def bind(self, **kwargs):
        return self

    def info(self, m):
        pass

    def warning(self, m):
        pass

    def error(self, m):
        pass

    def debug(self, m):
        pass


# ── Defect 1: no backoff between retries ────────────────────────────────

def test_retry_loop_does_not_hammer_with_zero_delay():
    """The retry loop must not fire all attempts in the same instant.

    A 403 is a server verdict, not a race — an immediate identical retry cannot
    succeed. Either back off between attempts, or do not retry at all.

    RED: the current loop has no delay, so all attempts land in ~0ms.
    """
    src = inspect.getsource(tts_base.TTSProviderBase.to_tts_stream)
    has_backoff = (
        "sleep" in src
        or "backoff" in src.lower()
        or "_RETRY" in src
    )
    assert has_backoff, (
        "to_tts_stream retries with NO delay — 5 identical attempts in the same "
        "instant against a 403 cannot succeed and only add latency "
        "(24s of silence observed in production 12:03:36-12:04:00)"
    )


def test_retry_budget_is_a_named_constant():
    """The retry count must be a named constant, not a magic 5 in the loop.

    Worst-case latency is then reviewable in one place instead of being an
    emergent property of a bare literal buried in the retry code.
    """
    src = inspect.getsource(tts_base.TTSProviderBase.to_tts_stream)
    has_named = any(
        tok in src for tok in ("_MAX_TTS_RETRIES", "_TTS_RETRIES", "_RETRY_")
    )
    assert has_named, (
        "the retry count is a bare magic number in to_tts_stream — the worst-case "
        "delay it implies is not reviewable"
    )


# ── Defect 2: the error signal must fire on BOTH exhaustion paths ────────

def test_emit_tts_error_is_called_on_every_retry_exhaustion_path():
    """Both branches of `to_tts` must notify the client on exhaustion.

    Found by reading the source: `_emit_tts_error` appears once (the
    delete_audio_file branch). The file branch reports to the log only, so a
    failure there leaves the client spinning forever.

    Counts the emit call sites against the number of exhaustion sites, so a
    future third branch cannot silently omit it.
    """
    src = inspect.getsource(tts_base.TTSProviderBase.to_tts_stream)

    n_exhaustion = src.count("语音生成失败: ")  # the error-level log at each site
    n_emit = src.count("_emit_tts_error(")

    assert n_exhaustion >= 2, (
        f"expected at least 2 exhaustion sites in to_tts, found {n_exhaustion} "
        "— this test's premise changed; re-read the source"
    )
    assert n_emit == n_exhaustion, (
        f"{n_exhaustion} retry-exhaustion sites but only {n_emit} "
        "`_emit_tts_error()` call(s): a failing branch reports to the log "
        "without telling the client, which waits forever for a terminal signal"
    )


def test_file_branch_emits_error_on_exhaustion():
    """Specifically: the non-delete_audio_file branch must emit.

    Asserted structurally (the branch body contains an emit) rather than by
    index, so unrelated edits do not make this brittle.
    """
    src = inspect.getsource(tts_base.TTSProviderBase.to_tts_stream)
    # The file branch is the one guarded by `else:` after delete_audio_file.
    marker = "tmp_file = self.generate_filename()"
    assert marker in src, "could not locate the file branch — re-read to_tts_stream"
    file_branch = src.split(marker, 1)[1]
    assert "_emit_tts_error(" in file_branch, (
        "the file branch of to_tts does not emit `tts error` on exhaustion — "
        "a client on this path waits forever (the v0.9.16 bug, still open here)"
    )


# ── The v0.9.16 contract must be preserved ──────────────────────────────

def test_emit_tts_error_is_not_tts_stop():
    """Regression guard: the error signal must never be a `tts stop`.

    `stop` means END-OF-REPLY; emitting it on a per-sentence failure made the
    client commit early and discard the remaining reply (the v0.9.15 "text is
    gone" regression, reverted).
    """
    src = inspect.getsource(tts_base.TTSProviderBase._emit_tts_error)
    assert '"stop"' not in src and "'stop'" not in src, (
        "_emit_tts_error mentions 'stop' — that is the reverted regression"
    )
    assert "error" in src, "_emit_tts_error no longer emits the error state"
