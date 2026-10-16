"""Tests for the voiceprint JSON wrap/unwrap contract (live code).

## Why this file exists

Deleting the completeness gate (v0.9.19) also deleted
`test_utterance_gate_integration.py`, which held the ONLY tests for the
`_wrap_content` / `_unwrap_content` JSON round-trip. But those methods are
**still live** on the answered path:

    core/providers/asr/base.py:211   self._unwrap_content(...)
    core/providers/asr/base.py:217   self._wrap_content(raw_text, safe_speaker)
    core/providers/asr/base.py:226   _build_enhanced_text  (the JSON wrap)
    core/providers/asr/base.py:235   _unwrap_content

So a live contract lost its coverage — a gap the deletion created, caught by
build-tester 2026-10-16. These tests close it.

## The contract

When voiceprint is active, `enhanced_text` reaching `handle_voice_stop` is

    {"speaker": "Vincent", "content": "講下今日天氣"}

and whatever reaches `startToChat` must be a SINGLE valid JSON object of that
shape. The historical bug was double-wrapping: concatenating two wrapped
payloads yields `{...}{...}`, which is not valid JSON and leaks literal braces
into the LLM prompt, the TTS output and the on-screen transcript.

These tests drive the REAL `handle_voice_stop` so the seam under test is the
production one, not a re-implementation.
"""

import asyncio
import json
import queue
import tempfile
from typing import Any, Optional
from unittest import mock

from core.providers.asr.base import ASRProviderBase
from core.providers.asr.dto.dto import InterfaceType


class _FakeLogger:
    def bind(self, **kwargs):
        return self

    def info(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass

    def debug(self, msg):
        pass


class _ConcreteASR(ASRProviderBase):
    """Minimal concrete subclass so the real `handle_voice_stop` can be called."""

    def __init__(self, transcript):
        self._transcript = transcript
        self._iface = InterfaceType.NON_STREAM
        self.output_dir = tempfile.mkdtemp(prefix="asr-wrap-test-")

    @property
    def interface_type(self):
        return self._iface

    @interface_type.setter
    def interface_type(self, value):
        self._iface = value

    async def speech_to_text_wrapper(self, pcm_data, session_id):
        return self._transcript, None

    async def speech_to_text(self, opus_data, session_id, artifacts=None):
        return self._transcript, None

    async def open_audio_channels(self, conn):
        pass

    async def close(self):
        pass

    def stop_ws_connection(self):
        pass


class _FakeVoiceprint:
    """Always identifies the same speaker — mirrors prod local_ecapa."""

    async def identify_speaker(self, wav_data, session_id):
        return "Vincent"


class _StubConn:
    """The slice of ConnectionHandler that `handle_voice_stop` actually touches.

    ⚠️ Every attribute here is read by the PRODUCTION path. A missing one raises
    AttributeError inside the handler, which its broad `except` swallows as
    `处理语音停止失败` — the test then sees `[]` and "passes" its negative
    assertions while the turn never ran (a vacuous green).
    """

    def __init__(self):
        # No `llm` needed any more: the judge is gone, so nothing calls it.
        self.logger = _FakeLogger()
        self.session_id = "sess-1"
        self.voiceprint_provider = _FakeVoiceprint()
        self.asr_audio = []
        self.read_config_from_api = True
        self.need_bind = False
        self.report_asr_enable = True
        self.chat_history_conf = 1
        self.report_queue = queue.Queue()
        self.device_id = "test-device"
        self.asr: Any = None
        self.introduced_speakers: set = set()
        self.current_speaker = None
        self.max_output_size = 0
        self.headers: dict = {}
        self.client_is_speaking = False
        self.client_listen_mode = "auto"
        self.config: dict = {}
        self.last_activity_time = 0.0
        self.close_after_chat = False
        self.client_abort = False
        self.executor: Any = None
        self.just_woken_up = False


def _run_fragment(conn, asr, text, answered):
    """Drive the REAL handle_voice_stop, capturing what startToChat receives."""
    asr._transcript = text

    with mock.patch(
        "core.providers.asr.base.startToChat",
        new=mock.AsyncMock(side_effect=lambda c, payload: answered.append(payload)),
    ):
        asyncio.run(asr.handle_voice_stop(conn, [b"\x00" * 40000]))


# ── The round-trip contract ──────────────────────────────────────────────

def test_wrapped_payload_is_one_valid_json_object():
    """THE regression contract: what reaches startToChat must be ONE valid JSON
    object — never two concatenated ones (`{...}{...}`)."""
    conn = _StubConn()
    asr = _ConcreteASR("")
    answered = []

    _run_fragment(conn, asr, "講下今日天氣。", answered)

    assert len(answered) == 1, f"expected 1 answer, got {len(answered)}: {answered}"
    payload = answered[0]
    parsed = json.loads(payload)  # raises on the `{...}{...}` corruption
    assert parsed["speaker"] == "Vincent"
    assert parsed["content"] == "講下今日天氣。"


def test_no_braces_leak_when_voiceprint_absent():
    """With no speaker identified the payload is bare text — no stray braces."""
    conn = _StubConn()
    conn.voiceprint_provider = None
    asr = _ConcreteASR("")
    answered = []

    _run_fragment(conn, asr, "講下今日天氣。", answered)

    assert len(answered) == 1, f"expected 1 answer, got {len(answered)}: {answered}"
    payload = answered[0]
    assert "{" not in payload and "}" not in payload, (
        f"braces leaked into a no-voiceprint payload: {payload!r}"
    )
    assert payload == "講下今日天氣。"


def test_utterance_is_answered_immediately_not_held():
    """The gate is gone: a complete utterance must be answered on the spot.

    This is the behavioural difference from v0.9.18, where an utterance could be
    held for seconds while the LLM judge ran — the latency that caused the
    "say it 3 times" loop.
    """
    conn = _StubConn()
    asr = _ConcreteASR("")
    answered = []

    _run_fragment(conn, asr, "講下今日天氣。", answered)

    assert len(answered) == 1, "utterance was not answered immediately"


def test_no_hold_attributes_are_referenced():
    """`_pending_utterance` must not be touched by the ASR path any more.

    A leftover reference would be a latent AttributeError swallowed by the
    handler's broad `except`.
    """
    import inspect

    import core.providers.asr.base as base

    src = inspect.getsource(base)
    assert "_pending_utterance" not in src, (
        "asr/base.py still references the removed hold attribute"
    )


# ── The helpers themselves (pure, cheap, worth pinning) ──────────────────

def test_unwrap_recovers_raw_text_from_wrapped_payload():
    """`_unwrap_content` must invert `_build_enhanced_text`."""
    asr = _ConcreteASR("")
    wrapped = asr._build_enhanced_text("我想問一個關於頭痛嘅", "Vincent")
    assert json.loads(wrapped)["content"] == "我想問一個關於頭痛嘅"
    assert asr._unwrap_content(wrapped) == "我想問一個關於頭痛嘅"


def test_unwrap_passes_through_bare_text():
    """Bare text (no voiceprint) must survive unwrap unchanged."""
    asr = _ConcreteASR("")
    assert asr._unwrap_content("講下今日天氣。") == "講下今日天氣。"
    assert asr._unwrap_content("") == ""
    assert asr._unwrap_content(None) == ""


def test_unwrap_tolerates_malformed_json():
    """A partial/garbled payload must not raise — it falls back to raw text."""
    asr = _ConcreteASR("")
    assert asr._unwrap_content('{"speaker": "Vincent"') == '{"speaker": "Vincent"'
    assert asr._unwrap_content("{}") == "{}"


def test_wrap_then_unwrap_round_trips_for_any_speaker():
    """Property: wrap→unwrap is identity on the content, for any speaker name."""
    asr = _ConcreteASR("")
    for speaker in ("Vincent", "Wayne", "Vivian", None, ""):
        for text in ("講下今日天氣。", "hello world", "我想話加個"):
            wrapped = asr._wrap_content(text, speaker)
            assert asr._unwrap_content(wrapped) == text, (
                f"round-trip failed for speaker={speaker!r} text={text!r}"
            )
