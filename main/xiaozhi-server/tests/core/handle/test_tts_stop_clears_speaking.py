"""Regression test: `tts stop` must clear `client_is_speaking` even when the
sentence_id has changed (a new turn started during the reply drain).

Bug (2026-10-03, present through v0.9.11): `send_tts_message(conn, "stop")`
guards BOTH `clearSpeakStatus()` and `audio_rate_controller.stop_sending()`
behind `if current_sentence_id != conn.sentence_id: return`. When the user
speaks a new utterance while the previous reply's audio is still draining, the
new turn bumps `conn.sentence_id`, so the stale stop hits that guard and returns
early — skipping `clearSpeakStatus()`. `conn.client_is_speaking` therefore stays
latched `True`.

The next genuine utterance then hits `startToChat`'s
`if conn.client_is_speaking: await handleAbortMessage(conn)` (receiveAudioHandle.py:87)
and SELF-ABORTS — the user's turn is dropped and they sit in "Thinking…" with no
reply.

The user's literal symptom: "seems it is taking a long time to reply… in
thinking…" (the turn was dropped, so no reply ever arrives).

Smoking gun from server.log (2026-10-03):
  20:21:10  大模型收到用户消息: 嘿聽日天氣如何？   (turn 1 starts)
  20:21:12  识别文本: 系.                          (trailing VAD re-fire)
  20:21:12  Abort message received                 (self-abort from stale flag)
  20:21:12  startToChat: client_abort set during intent routing — dropping the turn

The fix: `clearSpeakStatus()` must run UNCONDITIONALLY on `tts stop`, while the
`audio_rate_controller.stop_sending()` stays guarded (a stale stop must NOT
clobber a NEWER turn's in-flight audio queue — that is the reason the sentence_id
guard exists).
"""

import asyncio
from unittest import mock

from core.handle import sendAudioHandle


class _RateController:
    def __init__(self):
        self.stopped = False

    def stop_sending(self):
        self.stopped = True


class _StubConn:
    """Minimal ConnectionHandler stand-in for the `tts stop` path."""

    def __init__(self, sentence_id, client_is_speaking=True):
        self.sentence_id = sentence_id
        self.client_is_speaking = client_is_speaking
        self.session_id = "test-session"
        self.config = {"enable_stop_tts_notify": False}
        self.audio_rate_controller = _RateController()
        self.websocket = _FakeWebsocket()
        self.logger = _FakeLogger()

    def clearSpeakStatus(self):
        self.client_is_speaking = False


class _FakeWebsocket:
    def __init__(self):
        self.sent = []

    async def send(self, data):
        self.sent.append(data)


class _FakeLogger:
    def bind(self, **kwargs):
        return self

    def debug(self, msg):
        pass

    def info(self, msg):
        pass


def _run(coro):
    return asyncio.run(coro)


def test_tts_stop_clears_speaking_when_sentence_id_changed():
    """The user's bug: a stale `tts stop` (new turn already started) must STILL
    clear `client_is_speaking`, or the next turn self-aborts.

    Simulate the real sequence: the reply's audio is draining, and DURING that
    drain a new `startToChat` bumps `conn.sentence_id`. The stale stop then
    arrives with `current_sentence_id != conn.sentence_id` and must still reset
    the speaking flag."""

    conn = _StubConn(sentence_id="turn-2", client_is_speaking=True)

    # Patch _wait_for_audio_completion to simulate the new turn bumping the
    # sentence_id mid-drain (this is exactly what happens in the race).
    async def drain_with_new_turn(_conn):
        _conn.sentence_id = "turn-3"   # a newer turn started while draining

    with mock.patch.object(
        sendAudioHandle, "_wait_for_audio_completion", new=drain_with_new_turn
    ):
        _run(sendAudioHandle.send_tts_message(conn, "stop", None))

    assert conn.client_is_speaking is False, (
        "tts stop must clear client_is_speaking even when a new turn bumped the "
        "sentence_id — a stale True flag makes the next utterance self-abort "
        "(the 'long thinking / no reply' bug)"
    )


def test_tts_stop_does_not_stop_newer_turns_audio_controller():
    """The complementary invariant: a stale stop must NOT stop the rate
    controller of a NEWER turn. This is the reason the sentence_id guard exists
    and must be preserved — only client_is_speaking is safe to clear
    unconditionally."""

    conn = _StubConn(sentence_id="turn-2", client_is_speaking=True)

    async def drain_with_new_turn(_conn):
        _conn.sentence_id = "turn-3"

    with mock.patch.object(
        sendAudioHandle, "_wait_for_audio_completion", new=drain_with_new_turn
    ):
        _run(sendAudioHandle.send_tts_message(conn, "stop", None))

    assert conn.audio_rate_controller.stopped is False, (
        "a stale tts stop must NOT stop a newer turn's audio rate controller"
    )


def test_tts_stop_matching_sentence_clears_and_stops():
    """Normal path: when no new turn interleaved, both the speaking flag AND the
    rate controller are released."""

    conn = _StubConn(sentence_id="turn-2", client_is_speaking=True)

    async def drain_no_change(_conn):
        pass   # sentence_id unchanged

    with mock.patch.object(
        sendAudioHandle, "_wait_for_audio_completion", new=drain_no_change
    ):
        _run(sendAudioHandle.send_tts_message(conn, "stop", None))

    assert conn.client_is_speaking is False, "normal stop must clear the flag"
    assert conn.audio_rate_controller.stopped is True, "normal stop must stop the controller"
