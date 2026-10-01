"""Regression test for the interrupted-message re-fire bug in `handleAbortMessage`.

Bug (reported 2026-10-01, still present in v0.9.9): the v0.9.8 guard in
`startToChat` drops the interrupted turn at its await boundaries — but the abort
only clears TTS/report queues. It NEVER flushes the ASR/VAD audio buffers
(`reset_audio_states`), so half-buffered audio re-fires `handle_voice_stop` →
`startToChat` repeatedly for ONE spoken utterance. On the 2nd/3rd re-fire the
`client_abort` flag has already been cleared by the earlier drop, so the
interrupted message is answered anyway.

The user's literal symptom: "the interrupted message from me is queued into the
next round of Questions and Answers."

Smoking gun from server.log (2026-10-01 07:52):
  07:52:05  识别文本 其實我係想知道                              → answered
  07:52:10  Abort message received                               (queues cleared only)
  07:52:16  识别文本 …比特幣仲應唔應該走落去  → startToChat DROPS → flag=False
  07:52:30  识别文本 …比特幣系咪應該繼續走落去 → answered ❌ (flag already False)

Root fix under test: `handleAbortMessage` must also call
`conn.reset_audio_states()` so the buffered ASR audio (and VAD stop-trigger
state) is flushed on abort and cannot re-fire into `startToChat`.
"""

import asyncio
from unittest import mock

from core.handle import abortHandle


class _StubConn:
    """Minimal stand-in for ConnectionHandler — only what handleAbortMessage
    touches: close_after_chat, client_abort, clear_queues, reset_audio_states,
    clearSpeakStatus, websocket.send, logger, session_id."""

    def __init__(self):
        self.close_after_chat = False
        self.client_abort = False
        self.session_id = "test-session"
        self.queue_cleared = False
        self.audio_states_reset = False
        self.speak_status_cleared = False
        self.sent_messages = []
        self.logger = _FakeLogger()
        self.websocket = _FakeWebsocket(self.sent_messages)

    def clear_queues(self):
        self.queue_cleared = True

    def reset_audio_states(self):
        self.audio_states_reset = True

    def clearSpeakStatus(self):
        self.speak_status_cleared = True


class _FakeWebsocket:
    def __init__(self, sink):
        self._sink = sink

    async def send(self, data):
        self._sink.append(data)


class _FakeLogger:
    def bind(self, **kwargs):
        return self

    def info(self, msg):
        pass

    def debug(self, msg):
        pass


def _run(coro):
    return asyncio.run(coro)


def test_abort_flushes_audio_state_to_stop_re_fire():
    """Abort must flush ASR/VAD buffers (reset_audio_states) — otherwise a
    buffered half-utterance re-fires startToChat after the abort and the
    interrupted message is answered next round (the user's bug)."""
    conn = _StubConn()

    _run(abortHandle.handleAbortMessage(conn))

    assert conn.client_abort is True, "abort must set client_abort"
    assert conn.audio_states_reset is True, (
        "abort must call reset_audio_states() so buffered ASR audio cannot "
        "re-fire startToChat and answer the interrupted message"
    )


def test_abort_still_clears_queues_and_speech_status():
    """The existing abort behaviour (clear TTS/report queues + speak status +
    tts-stop send) must be preserved — don't regress the complementary path."""
    conn = _StubConn()

    _run(abortHandle.handleAbortMessage(conn))

    assert conn.queue_cleared is True, "abort must still clear TTS/report queues"
    assert conn.speak_status_cleared is True, "abort must still clear speak status"
    assert len(conn.sent_messages) == 1, "abort must still send the tts stop"
    assert '"state": "stop"' in conn.sent_messages[0], "sent message must be the tts stop"
