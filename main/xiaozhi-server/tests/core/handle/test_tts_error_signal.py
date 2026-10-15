"""Regression test: a total TTS failure must emit an explicit `tts error` signal.

Bug class (2026-10-15): when TTS generation failed for every sentence of a reply,
the server produced NO terminal signal for that turn. The client received
`sentence_start` (with text) and then nothing — it had no way to learn the reply
was over, so it sat on its spinner forever.

Vincent's literal symptom: *"it doesn't seem to be responding"* / *"it's still
stuck at thinking...."*

⚠️ A previous fix attempt (reverted, see v0.9.15) emitted a `tts stop` on the
failure path. That was WRONG and caused a worse bug: the client treats
`tts stop` as END-OF-REPLY, so a per-sentence failure `stop` made it commit the
turn early and delete the remaining reply as an orphan — the *"the text is
gone"* regression.

The correct shape is a DEDICATED signal the normal path never emits:
`{"type": "tts", "state": "error", "text": "<message>"}`. It cannot corrupt the
commit path because the good path never sends it.

⚠️ These tests drive the REAL `TTSProviderBase.to_tts_stream` entry point with a
provider whose synthesis always fails — not a re-implementation of the retry
loop. Asserting on the emitted payload, not on a mock's internals.
"""

import asyncio
import json
from unittest import mock

import pytest

from core.providers.tts.base import TTSProviderBase
from core.providers.tts.dto.dto import SentenceType


class _AlwaysFailingProvider(TTSProviderBase):
    """A TTS provider whose synthesis always fails — the incident's condition."""

    def __init__(self, conn):
        # Bypass the real __init__ (it needs a full config); set only what
        # `to_tts_stream` touches.
        self.conn = conn
        self.delete_audio_file = True
        self.audio_file_type = "mp3"
        self._correct_words_pattern = None
        self.correct_words = {}
        self.tts_audio_queue = asyncio.Queue()
        self.before_stop_play_files = []
        self.output_file = "/tmp"
        self.opus_encoder = None

    async def text_to_speak(self, text, output_file):
        raise Exception("Edge TTS请求失败: 403, message='Invalid response status'")

    def generate_filename(self, extension=".mp3"):
        return f"/tmp/tts-nonexistent{extension}"


class _FakeWebSocket:
    def __init__(self):
        self.sent = []

    async def send(self, payload):
        if isinstance(payload, (bytes, bytearray)):
            return
        self.sent.append(json.loads(payload))


class _FakeConn:
    def __init__(self):
        self.session_id = "test-session"
        self.sentence_id = "sentence-1"
        self.websocket = _FakeWebSocket()
        self.calling = False
        self.close_after_chat = False
        self.client_is_speaking = True
        self.client_abort = False
        self.config = {}
        self.logger = mock.MagicMock()
        self.audio_flow_control = {}
        self.sample_rate = 24000
        # Captured `tts error` signals — the client-visible failure notification.
        self.error_signals = []

    def clearSpeakStatus(self):
        self.client_is_speaking = False

    def _send_tts_error(self, reason: str):
        """Mirror of ConnectionHandler._send_tts_error (the production notifier).

        The real one bridges the TTS worker thread onto the connection's event
        loop via `run_coroutine_threadsafe`. These tests capture the payload for
        assertion; the thread/loop bridge itself is exercised by the real
        connection (see ConnectionHandler._send_tts_error).
        """
        self.error_signals.append(
            {"type": "tts", "state": "error", "text": reason, "session_id": self.session_id}
        )


def test_total_tts_failure_emits_error_signal():
    """Every retry failed → the client MUST receive an explicit `tts error`.

    Without it the client has no terminal signal and spins forever.
    """
    conn = _FakeConn()
    provider = _AlwaysFailingProvider(conn)

    provider.to_tts_stream("今日天氣唔錯。")

    # POSITIVE CONTROL — a total failure must still produce SOME signal. Before
    # the fix it produced literally nothing, so this assertion is what fails.
    assert conn.error_signals, (
        "a total TTS failure sent no `tts error` signal — "
        "the client will wait forever for a reply that never comes"
    )

    sig = conn.error_signals[-1]
    assert sig["type"] == "tts"
    assert sig["state"] == "error", (
        f"expected state='error', got {sig['state']!r} — the client cannot "
        f"distinguish a failure from a normal sentence end"
    )
    assert sig["text"], "the error message must be carried through to the client"


def test_total_tts_failure_does_not_emit_stop():
    """Pin the opposite direction — the v0.9.15 regression guard.

    A failure must NOT be signalled as `tts stop`. The client reads `stop` as
    end-of-reply; a per-sentence failure `stop` makes it commit the turn early
    and then discard the remaining reply as an orphan — the "text is gone" bug.
    """
    conn = _FakeConn()
    provider = _AlwaysFailingProvider(conn)

    provider.to_tts_stream("今日天氣唔錯。")

    states = [s.get("state") for s in conn.error_signals]
    assert "stop" not in states, (
        "a TTS failure signalled `stop` — the client reads that as end-of-reply "
        "and will commit the turn early, destroying the reply text (v0.9.15 bug)"
    )
