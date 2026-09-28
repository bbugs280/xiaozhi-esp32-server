"""Regression tests for the interrupted-turn bug in `startToChat`.

Bug (2026-09-28, v0.9.8): when the user taps STOP / barges in while a spoken
utterance is being routed through `handle_user_intent` (which awaits an LLM
call), the abort sets `conn.client_abort = True` on the event loop — but
`startToChat` then resumed and UNCONDITIONALLY reset `client_abort = False`
before `executor.submit(conn.chat, ...)`, wiping the abort and answering the
interrupted utterance as a fresh turn next round.

The user's literal symptom: "the interrupted message from me is queued into the
next round of Questions and Answers."
"""

import asyncio
from unittest import mock

import pytest

from core.handle import receiveAudioHandle


class _StubConn:
    """Minimal stand-in for ConnectionHandler — only the attributes
    `startToChat` actually touches on the non-bind, non-rate-limited path."""

    def __init__(self, client_abort_init=False):
        self.need_bind = False
        self.max_output_size = 0
        self.headers = {}
        self.client_is_speaking = False
        self.client_listen_mode = "auto"
        self.client_abort = client_abort_init
        self.introduced_speakers = set()
        self.current_speaker = None
        # The executor.submit target is captured so the test can assert whether
        # the turn was (or was NOT) answered.
        self.submitted_chats = []
        self.executor = _FakeExecutor(self.submitted_chats)
        # `conn.chat` is the method reference `executor.submit` binds to; it is
        # captured via the fake executor above, but must exist as an attribute
        # so the (buggy) code under test can call `conn.chat` without crashing.
        self.chat = "chat-method"
        self.logger = _FakeLogger()


class _FakeExecutor:
    def __init__(self, sink):
        self._sink = sink

    def submit(self, fn, *args):
        self._sink.append(args[0] if args else None)
        return None


class _FakeLogger:
    def bind(self, **kwargs):
        return self

    def info(self, msg):
        pass

    def debug(self, msg):
        pass

    def error(self, msg):
        pass


def _run(coro):
    return asyncio.run(coro)


def test_normal_turn_is_answered():
    """A turn with no abort in flight must still reach conn.chat."""
    conn = _StubConn(client_abort_init=False)
    with mock.patch.object(
        receiveAudioHandle, "handle_user_intent", new=mock.AsyncMock(return_value=False)
    ), mock.patch.object(
        receiveAudioHandle, "send_stt_message", new=mock.AsyncMock()
    ):
        _run(receiveAudioHandle.startToChat(conn, "hello"))

    assert conn.submitted_chats == ["hello"], "a clean turn must be answered"


def test_interrupted_turn_is_not_answered():
    """An abort landing during intent routing must DROP the turn — this is the
    user's literal bug: the interrupted message must NOT be answered next round."""
    conn = _StubConn(client_abort_init=False)

    async def intent_with_abort(_conn, _text):
        # Simulate the user barging in while handle_user_intent is awaiting.
        _conn.client_abort = True
        return False

    with mock.patch.object(
        receiveAudioHandle, "handle_user_intent", new=intent_with_abort
    ), mock.patch.object(
        receiveAudioHandle, "send_stt_message", new=mock.AsyncMock()
    ):
        _run(receiveAudioHandle.startToChat(conn, "interrupted message"))

    assert conn.submitted_chats == [], (
        "an interrupted turn must NOT be answered (the 'queued into next round' bug)"
    )


def test_interrupted_turn_does_not_leak_abort_flag():
    """Dropping a turn must clear client_abort, or the NEXT genuine turn is
    falsely dropped too (build-tester's flag-leak finding)."""
    conn = _StubConn(client_abort_init=True)

    with mock.patch.object(
        receiveAudioHandle, "handle_user_intent", new=mock.AsyncMock(return_value=False)
    ), mock.patch.object(
        receiveAudioHandle, "send_stt_message", new=mock.AsyncMock()
    ):
        _run(receiveAudioHandle.startToChat(conn, "dropped turn"))

    # After the drop, the flag must be clear so a subsequent turn proceeds.
    assert conn.client_abort is False, "drop must not leave client_abort latched"

    # And a follow-up turn with the flag now clear must be answered.
    conn.submitted_chats.clear()
    with mock.patch.object(
        receiveAudioHandle, "handle_user_intent", new=mock.AsyncMock(return_value=False)
    ), mock.patch.object(
        receiveAudioHandle, "send_stt_message", new=mock.AsyncMock()
    ):
        _run(receiveAudioHandle.startToChat(conn, "next question"))


@pytest.mark.parametrize("abort_at", ["intent", "stt"])
def test_abort_at_either_await_boundary_drops_turn(abort_at):
    """The fix guards BOTH await boundaries (handle_user_intent and
    send_stt_message). An abort at either must drop the turn."""
    conn = _StubConn(client_abort_init=False)

    async def intent(_conn, _text):
        if abort_at == "intent":
            _conn.client_abort = True
        return False

    async def stt(_conn, _text):
        # send_stt_message runs only when the intent guard did NOT trip.
        if abort_at == "stt":
            _conn.client_abort = True

    with mock.patch.object(receiveAudioHandle, "handle_user_intent", new=intent), \
         mock.patch.object(receiveAudioHandle, "send_stt_message", new=stt):
        _run(receiveAudioHandle.startToChat(conn, "msg"))

    assert conn.submitted_chats == [], "turn must be dropped when abort lands at %r" % abort_at
