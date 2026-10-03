"""Regression test: `listen "start"` must clear the `client_abort` flag.

Bug (reported 2026-10-03, present through v0.9.10): after a STOP-interrupt
(`abort` sets `conn.client_abort = True`), the next tap sends `listen start`,
which calls `reset_audio_states()` (clears VAD/ASR audio) but does NOT clear
`client_abort`. The flag therefore stays latched `True` into the next turn, so
the user's next genuine spoken utterance hits the guard in `startToChat` and is
silently dropped.

The user's literal symptom: "after I click stop to interrupt, it takes a while
to resume" — the first re-speak after a STOP is discarded, so the user must
speak a second time before the assistant hears them.

Smoking gun from server.log (2026-10-03):
  10:53:16  Abort message received  → client_abort=True
  10:53:18  listen start            → reset_audio_states() only (flag STILL True)
  10:53:31  识别文本: <new utterance>
  10:53:31  startToChat: client_abort set during intent routing — dropping the turn

The upstream `startToChat` tests cover the guard dropping an aborted turn, and
`test_abort_handle.py` covers the abort flushing audio state — but neither model
the `listen start` boundary, which is where the flag MUST be cleared for the
NEXT turn to be heard. This test fills that gap.
"""

import asyncio

from core.handle.textHandler.listenMessageHandler import ListenTextMessageHandler
from core.handle.textMessageType import TextMessageType


class _StubConn:
    """Minimal stand-in for ConnectionHandler — only what the `listen "start"`
    branch touches: client_listen_mode, reset_audio_states(), and logger."""

    def __init__(self, client_abort_init):
        self.client_listen_mode = "auto"
        self.client_abort = client_abort_init
        self.audio_states_reset = False
        self.logger = _FakeLogger()

    def reset_audio_states(self):
        self.audio_states_reset = True


class _FakeLogger:
    def bind(self, **kwargs):
        return self

    def debug(self, msg):
        pass

    def info(self, msg):
        pass


def _run(coro):
    return asyncio.run(coro)


def test_listen_start_clears_client_abort():
    """A `listen start` (the user's tap after a STOP) must clear `client_abort`
    so the next spoken utterance is NOT dropped by the startToChat guard.

    This is the user's bug: after interrupting with STOP, the first re-speak is
    silently discarded because the abort flag is still latched True."""
    # Simulate the real sequence: an abort left the flag latched True, then the
    # user taps again → the client sends `listen start`.
    conn = _StubConn(client_abort_init=True)

    handler = ListenTextMessageHandler()
    _run(handler.handle(conn, {"type": "listen", "state": "start", "mode": "auto"}))

    assert conn.audio_states_reset is True, (
        "listen start must still reset audio states (existing behaviour)"
    )
    assert conn.client_abort is False, (
        "listen start must clear the latched client_abort flag, or the next "
        "genuine utterance is silently dropped (the 'takes a while to resume' bug)"
    )


def test_listen_start_without_prior_abort_stays_clear():
    """Idempotence: a `listen start` with no prior abort must not corrupt state."""
    conn = _StubConn(client_abort_init=False)

    handler = ListenTextMessageHandler()
    _run(handler.handle(conn, {"type": "listen", "state": "start", "mode": "auto"}))

    assert conn.client_abort is False, "a clean listen start must leave client_abort clear"
