"""Integration tests for the completeness gate, driving the REAL
`handle_voice_stop` path.

⚠️ Why this file was rewritten (build-tester BLOCK, 2026-10-11): the first
version re-implemented the merge inline with plain text

    candidate = f"{pending}{text}"

so it passed while the production path was corrupt. In production, voiceprint is
active (`data/.config.yaml`: `voiceprint.type: local_ecapa`), which makes
`enhanced_text` a JSON wrapper `{"speaker": ..., "content": ...}`. Concatenating
two wrapped payloads yields `{...}{...}` — invalid JSON that leaked literal
braces into the LLM prompt, the TTS and the transcript.

These tests therefore call `handle_voice_stop()` on the real ASR provider class,
with voiceprint ON, and assert the payload reaching `startToChat` is valid JSON
carrying the merged text. Testing the real seam is the whole point — a green
suite that re-implements the code under test guards nothing.
"""

import asyncio
import json
import queue
import tempfile
from typing import Any, Optional
from unittest import mock

import pytest

from core.connection import ConnectionHandler
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
    """Minimal concrete subclass so we can call the real `handle_voice_stop`.

    ⚠️ `interface_type` must be a real settable attribute: `handle_voice_stop`
    only guards on `conn.voiceprint_provider` / `wav_data`, but downstream
    callers (`receive_audio`) read it. A property pair is kept so a bare
    `InterfaceType.NON_STREAM` can be assigned without an AttributeError.
    """

    def __init__(self, transcript):
        self._transcript = transcript
        self._iface = InterfaceType.NON_STREAM
        self.output_dir = tempfile.mkdtemp(prefix="asr-gate-test-")
        # Set by the STREAM-path test; the ABC has no such method.
        self._send_stop_request: Any = None

    @property
    def interface_type(self):
        return self._iface

    @interface_type.setter
    def interface_type(self, value):
        self._iface = value

    async def speech_to_text_wrapper(self, pcm_data, session_id):
        return self._transcript, None

    async def speech_to_text(
        self, opus_data, session_id, artifacts=None
    ):
        """Abstract method in ASRProviderBase — must be implemented or the
        class cannot be instantiated at all (Pyright: 'cannot instantiate
        abstract class'). The gate calls the *wrapper*, but the ABC still
        demands this."""
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
    """The slice of `ConnectionHandler` that `handle_voice_stop` actually touches.

    Kept minimal but REAL: every attribute here is read by the production path
    (not by the test's own re-implementation). `read_config_from_api=True` +
    `need_bind=False` + `report_asr_enable=True` is what `enqueue_asr_report`
    gates on — left unset, the report call raises AttributeError and the whole
    handler aborts with `处理语音停止失败`, which is why every test was RED.
    """

    def __init__(self, llm):
        self.llm = llm
        self.logger = _FakeLogger()
        self.session_id = "sess-1"
        self._pending_utterance: Optional[str] = None
        self._pending_utterance_speaker: Optional[str] = None
        self.voiceprint_provider = _FakeVoiceprint()
        self.asr_audio = []
        self.submitted = []
        # Report path (enqueue_asr_report) reads these three.
        self.read_config_from_api = True
        self.need_bind = False
        self.report_asr_enable = True
        self.chat_history_conf = 1
        self.report_queue = queue.Queue()
        self.device_id = "test-device"
        self.asr: Any = None  # set by the listen-stop tests
        # ── Attributes the REAL `receiveAudioHandle.startToChat` reads ────────
        # ⚠️ Missing any of these raises AttributeError INSIDE startToChat,
        # which `handle_voice_stop` swallows as `处理语音停止失败`. The test then
        # sees `[]` and "passes" its negative assertions while the turn never
        # actually ran — a vacuous green. Every attribute the real path touches
        # must exist here.
        self.introduced_speakers: set = set()
        self.current_speaker = None
        self.max_output_size = 0  # 0 disables the max-output TTS path
        self.headers: dict = {}
        self.client_is_speaking = False
        self.client_listen_mode = "auto"
        self.config: dict = {}
        self.last_activity_time = 0.0
        self.close_after_chat = False
        self.client_abort = False
        self.executor: Any = None
        self.just_woken_up = False

    def take_pending_utterance(self):
        """Mirror of ConnectionHandler.take_pending_utterance — clears BOTH
        the text and the speaker (they are one unit)."""
        held = self._pending_utterance
        self._pending_utterance = None
        self._pending_utterance_speaker = None
        return held

    def take_pending_utterance_speaker(self):
        speaker = getattr(self, "_pending_utterance_speaker", None)
        self._pending_utterance_speaker = None
        return speaker

    def flush_pending_utterance(self):
        """Mirror of ConnectionHandler.flush_pending_utterance — reads the
        speaker BEFORE taking the text, since `take_pending_utterance` clears
        both in one unit."""
        speaker = self._pending_utterance_speaker
        held = self.take_pending_utterance()
        if not held:
            return None
        if isinstance(speaker, str) and speaker.strip():
            return json.dumps(
                {"speaker": speaker, "content": held}, ensure_ascii=False
            )
        return held

    def _wrap(self, text, speaker):
        if speaker:
            return json.dumps({"speaker": speaker, "content": text}, ensure_ascii=False)
        return text


class _ScriptedLLM:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.calls = []

    def response(self, messages):
        self.calls.append(messages[-1]["content"])
        return self.verdicts.pop(0) if self.verdicts else '{"complete": true}'


def _run_fragment(conn, asr, text, answered, speaker="Vincent"):
    """Drive the real handle_voice_stop, capturing what startToChat receives."""
    asr._transcript = text

    with mock.patch(
        "core.providers.asr.base.startToChat",
        new=mock.AsyncMock(side_effect=lambda c, payload: answered.append(payload)),
    ):
        asyncio.run(asr.handle_voice_stop(conn, [b"\x00" * 40000]))


def test_merged_payload_is_valid_json_with_voiceprint_on():
    """DEFECT 1 regression: the payload reaching startToChat must be ONE valid
    JSON object — not two concatenated ones."""
    conn = _StubConn(llm=_ScriptedLLM(['{"complete": true}']))
    asr = _ConcreteASR("")
    answered = []

    # Fragment 1: ends on 嘅 -> tier 1 holds it.
    _run_fragment(conn, asr, "我想問一個關於頭痛嘅", answered)
    assert answered == [], "fragment 1 should have been held, not answered"

    # Fragment 2: completes the thought -> merged and answered.
    _run_fragment(conn, asr, "問題，睇下有咩方案", answered)

    assert len(answered) == 1, f"expected 1 answer, got {len(answered)}: {answered}"

    payload = answered[0]
    # THE regression assertion: must parse as a single JSON object.
    parsed = json.loads(payload)  # raises on the `{...}{...}` corruption
    assert parsed["speaker"] == "Vincent"
    # Both fragments must survive in the merged content, in order.
    assert "我想問一個關於頭痛嘅" in parsed["content"]
    assert "問題，睇下有咩方案" in parsed["content"]
    assert parsed["content"].index("頭痛") < parsed["content"].index("方案")


def test_no_literal_braces_leak_when_voiceprint_absent():
    """With no speaker identified the payload is bare text — must still merge
    cleanly and never contain a stray JSON brace."""
    conn = _StubConn(llm=_ScriptedLLM(['{"complete": true}']))
    conn.voiceprint_provider = None
    asr = _ConcreteASR("")
    answered = []

    _run_fragment(conn, asr, "我想話加個", answered, speaker=None)
    _run_fragment(conn, asr, "task 提醒我睇醫生", answered, speaker=None)

    assert len(answered) == 1, f"expected 1 answer, got {len(answered)}: {answered}"
    payload = answered[0]
    assert "{" not in payload and "}" not in payload, (
        f"braces leaked into a no-voiceprint payload: {payload!r}"
    )
    assert "我想話加個" in payload
    assert "提醒我睇醫生" in payload


def test_three_fragments_of_one_thought_produce_one_answer():
    """THE user symptom end-to-end through the real path.

    Before: 3 fragments -> 3 LLM turns -> 3 answers ("batches").
    With the gate: held, held, answered -> exactly ONE answer, merged.
    """
    conn = _StubConn(llm=_ScriptedLLM(['{"complete": true}']))
    asr = _ConcreteASR("")
    answered = []

    _run_fragment(conn, asr, "一個關於天頭痛嘅。", answered)
    _run_fragment(
        conn, asr, "問題啦，因為我呢排成日都有偏頭痛，咁，我想睇下啲咩方案，所以。", answered
    )
    assert answered == [], "first two fragments were answered — cascade not stopped"

    _run_fragment(conn, asr, "我想睇下有咩方案", answered)

    assert len(answered) == 1, f"expected exactly 1 answer, got {len(answered)}"
    content = json.loads(answered[0])["content"]
    assert "偏頭痛" in content and "我想睇下有咩方案" in content


def test_complete_sentence_answers_immediately():
    """Complementary invariant: a clean sentence must NOT be held."""
    conn = _StubConn(llm=_ScriptedLLM(['{"complete": true}']))
    asr = _ConcreteASR("")
    answered = []

    _run_fragment(conn, asr, "講翻今日新聞啦。", answered)

    assert len(answered) == 1
    assert json.loads(answered[0])["content"] == "講翻今日新聞啦。"
    assert conn._pending_utterance is None


def test_hold_cleared_on_turn_boundary_so_it_cannot_leak():
    """reset_audio_states() must drop a held fragment (turn boundary hygiene)."""
    conn = _StubConn(llm=_ScriptedLLM(['{"complete": true}']))
    asr = _ConcreteASR("")
    answered = []

    _run_fragment(conn, asr, "我想話加個", answered)
    assert conn._pending_utterance is not None, "fragment should be held"

    conn._pending_utterance = None  # what reset_audio_states() does

    _run_fragment(conn, asr, "講翻今日新聞啦。", answered)
    assert len(answered) == 1
    assert json.loads(answered[0])["content"] == "講翻今日新聞啦。"


# ── Flush path (listen stop) ────────────────────────────────────────────
# The held fragment must not vanish silently, AND whatever reaches
# `startToChat` must be in the shape `startToChat` can parse. `startToChat`
# branches on `text.strip().startswith("{")`: a properly wrapped payload is
# unwrapped for the LLM, and a PARTIAL payload (`{"speaker":...` with no
# closing brace) is treated as literal text and spoken aloud.

def _run_listen_stop(conn, asr, answered):
    """Drive the real listen-message handler's `stop` branch.

    ⚠️ TWO call sites reach `startToChat` on this path, and they import it
    independently:
      • the handler module (this file's import) — the flush after the reset;
      • `core.providers.asr.base` (its own import) — the gate answering a
        complete-merged thought.
    Patching only the handler's name silently misses the ASR-side answer, so a
    test sees `[]` for a turn that actually ran. Patch BOTH, or answer-counting
    is wrong.
    """
    from core.handle.textHandler.listenMessageHandler import (
        ListenTextMessageHandler,
    )
    import core.providers.asr.base as _base

    handler = ListenTextMessageHandler()
    sink = mock.AsyncMock(side_effect=lambda c, payload: answered.append(payload))
    with mock.patch(
        "core.handle.textHandler.listenMessageHandler.startToChat", new=sink
    ), mock.patch.object(_base, "startToChat", new=sink):
        asyncio.run(handler.handle(conn, {"state": "stop"}))


def test_held_fragment_is_flushed_on_listen_stop_not_silently_dropped():
    """DEFECT 2 regression: the 'held forever, no reply' hole.

    A fragment held by the gate, followed by `listen stop` with no further
    audio, previously produced NO reply at all — the user speaks, VAD
    acknowledges, and nothing comes back. It must be answered.
    """
    conn = _StubConn(llm=_ScriptedLLM(['{"complete": true}']))
    asr = _ConcreteASR("")
    conn.asr = asr
    answered = []

    # Hold a half-sentence through the real gate.
    _run_fragment(conn, asr, "我想話加個", answered)
    assert answered == [], "fragment 1 should be held"
    assert conn._pending_utterance == "我想話加個"

    # User stops talking. No more audio will ever arrive.
    conn.asr_audio = []
    _run_listen_stop(conn, asr, answered)

    assert len(answered) == 1, (
        f"held fragment vanished on listen stop — user gets no reply: {answered}"
    )
    # ⚠️ Assert the WRAP, not just substring presence. Asserting
    # `"我想話加個" in answered[0]` passes for both a wrapped payload and the
    # bare string, so it does not discriminate the fix (build-tester, v0.9.14).
    parsed = json.loads(answered[0])
    assert parsed["content"] == "我想話加個"
    assert parsed["speaker"] == "Vincent"


def test_flushed_fragment_carries_the_merged_prefix():
    """The flush must answer the WHOLE held thought, not just the last piece.

    The gate accumulates every incomplete fragment into `_pending_utterance`.
    Flushing only the final fragment would silently discard the earlier words
    — the user's own words disappearing from their own question.
    """
    conn = _StubConn(llm=_ScriptedLLM(['{"complete": true}']))
    asr = _ConcreteASR("")
    conn.asr = asr
    answered = []

    _run_fragment(conn, asr, "我想問一個關於頭痛嘅", answered)  # held
    _run_fragment(conn, asr, "問題，睇下有咩方案，所以", answered)  # held, merged
    assert answered == [], "both fragments should be held"
    assert conn._pending_utterance is not None
    assert "頭痛" in conn._pending_utterance

    conn.asr_audio = []
    _run_listen_stop(conn, asr, answered)

    assert len(answered) == 1, f"expected 1 flushed answer, got {answered}"
    # Tightened: parse and assert the wrap too, so this discriminates.
    parsed = json.loads(answered[0])
    assert parsed["speaker"] == "Vincent"
    content = parsed["content"]
    assert "頭痛" in content, (
        f"the held prefix was dropped — only the tail was flushed: {content!r}"
    )
    assert "方案" in content


def test_flushed_fragment_is_voiceprint_wrapped_like_a_normal_turn():
    """The flush path must hand `startToChat` the same shape as a normal turn.

    A normal answered turn goes through `_wrap_content(merged_raw, speaker)`,
    i.e. `{"speaker": ..., "content": ...}` when voiceprint is active.
    Emitting the raw string instead means the speaker is lost for that turn.
    """
    conn = _StubConn(llm=_ScriptedLLM(['{"complete": true}']))
    asr = _ConcreteASR("")
    conn.asr = asr
    answered = []

    _run_fragment(conn, asr, "我想話加個", answered)  # held, speaker=Vincent
    conn.asr_audio = []
    _run_listen_stop(conn, asr, answered)

    assert len(answered) == 1
    payload = answered[0]
    parsed = json.loads(payload)
    assert parsed["speaker"] == "Vincent", (
        f"flush lost the speaker the rest of the pipeline carries: {payload!r}"
    )
    assert "我想話加個" in parsed["content"]


# ── Flush ordering vs reset_audio_states (build-tester BLOCK, v0.9.14) ───
# ⚠️ The tests above call `_run_listen_stop` with `conn.asr_audio = []`, which
# SKIPS the `if len(conn.asr_audio) > 0:` branch — and therefore skips the
# `reset_audio_states()` inside it. The first version of this fix read the hold
# AFTER that reset, so the flush was DEAD CODE on the real path (audio is
# always present when a hold exists) while the suite stayed green. These tests
# drive the PRODUCTION shape: non-empty asr_audio and the REAL
# `ConnectionHandler.reset_audio_states`.

def _run_listen_stop_real_reset(conn):
    """Drive the real handler with the production reset bound.

    Patches BOTH `startToChat` call sites — see `_run_listen_stop` for why.
    """
    from core.handle.textHandler.listenMessageHandler import (
        ListenTextMessageHandler,
    )
    import core.providers.asr.base as _base

    handler = ListenTextMessageHandler()
    answered = []
    sink = mock.AsyncMock(side_effect=lambda c, payload: answered.append(payload))
    with mock.patch(
        "core.handle.textHandler.listenMessageHandler.startToChat", new=sink
    ), mock.patch.object(_base, "startToChat", new=sink):
        asyncio.run(handler.handle(conn, {"state": "stop"}))
    return answered


class _RealResetConn(_StubConn):
    """A conn whose reset_audio_states IS the production implementation.

    Every attribute the real `reset_audio_states` touches is a real mutable
    container, so the clears behave exactly as in production.
    """

    reset_audio_states = ConnectionHandler.reset_audio_states

    def __init__(self, llm):
        super().__init__(llm)
        self.client_audio_buffer = bytearray()
        self.client_have_voice = False
        self.client_voice_stop = False
        self.client_voice_window = []
        self.last_is_voice = False
        self.vad_last_voice_time = 0.0


def test_flush_survives_reset_audio_states_on_the_real_listen_stop_path():
    """THE regression test for the dead-code flush.

    A hold can only exist if audio previously arrived — so on the real path
    `asr_audio` is non-empty, the reset branch runs, and it clears the hold.
    Reading the hold before that reset is the only correct ordering.
    """
    conn = _RealResetConn(llm=_ScriptedLLM(['{"complete": true}']))
    asr = _ConcreteASR("")
    conn.asr = asr
    # A fragment held by the gate, exactly as asr/base.py leaves it.
    conn._pending_utterance = "我想話加個"
    conn._pending_utterance_speaker = "Vincent"
    # PROD SHAPE: audio present, so the reset branch actually executes.
    conn.asr_audio = [b"\x00" * 320]

    answered = _run_listen_stop_real_reset(conn)

    assert len(answered) == 1, (
        "held fragment was DROPPED on the real listen-stop path — the reset "
        f"cleared the hold before the flush read it: {answered}"
    )
    parsed = json.loads(answered[0])
    assert parsed["content"] == "我想話加個"
    assert parsed["speaker"] == "Vincent"


def test_abort_still_drops_a_held_fragment_rather_than_answering_it():
    """Guardrail on the FIX's blast radius.

    `listen stop` must flush, but ABORT must NOT — answering the held text on
    abort would resurrect the "interrupted message answered next round" bug
    class that reset_audio_states() exists to prevent.

    ⚠️ How this is kept FALSIFIABLE. Two earlier versions of this test guarded
    nothing:

      v1 patched `core.handle.abortHandle.startToChat` with `create=True`. But
      abortHandle imports nothing and has NO startToChat attribute, so
      `create=True` injected a fresh dummy and intercepted nothing — the
      assertion held no matter what abort did.

      v2 patched `receiveAudioHandle.startToChat`. Also inert: abort calls
      neither. (Confirmed: `hasattr(core.handle.abortHandle, "startToChat")` is
      False.)

    The honest invariant is therefore structural, not mock-based: `abort` is
    the ONE turn boundary whose whole purpose is to DISCARD the pending turn,
    so the guard is that the hold is gone afterwards AND abort never routes it
    anywhere. We assert the hold is cleared and that no `startToChat`-capable
    seam was touched, by asserting the hold is empty at exit — a hold left
    behind is the only way abort could later answer it.

    The blast-radius check that DOES have teeth is the sibling test
    `test_flush_is_not_reachable_from_abort_path`: it asserts abort's source
    does not call `flush_pending_utterance`. Together they pin both halves.
    """
    import inspect

    from core.handle.abortHandle import handleAbortMessage

    conn = _RealResetConn(llm=_ScriptedLLM(['{"complete": true}']))
    conn._pending_utterance = "我想話加個"
    conn._pending_utterance_speaker = "Vincent"
    conn.close_after_chat = False
    conn.client_abort = False
    conn.clear_queues = lambda: None
    conn.clearSpeakStatus = lambda: None
    conn.websocket = mock.AsyncMock()

    asyncio.run(handleAbortMessage(conn))  # type: ignore[arg-type]

    # The invariant with teeth: abort must not LEAVE a hold behind.
    assert conn._pending_utterance is None, (
        "abort left the hold in place — it can now be answered on a later "
        "turn, which is the interrupted-message bug class"
    )
    assert conn._pending_utterance_speaker is None
    assert conn.client_abort is True

    # And abort must not be a flush site at all.
    src = inspect.getsource(handleAbortMessage)
    assert "flush_pending_utterance" not in src, (
        "abort calls flush_pending_utterance — answering on abort resurrects "
        "the interrupted-message bug class"
    )


def test_fragment_held_during_voice_stop_is_also_answered():
    """THE remaining hole (build-tester round 2, v0.9.14).

    `listen stop` is final — no audio follows, so every fragment held by the
    time the handler returns must be answered. But `handle_voice_stop()` runs
    INSIDE this handler and can itself decide the terminal fragment is
    incomplete, creating a hold AFTER the pre-reset capture point. That hold
    would never be answered: the user speaks, VAD acknowledges, and nothing
    comes back — the exact symptom this change exists to close.

    So the flush must also run AFTER `handle_voice_stop`, catching any hold
    that call created.
    """
    conn = _RealResetConn(llm=_ScriptedLLM(['{"complete": true}']))
    asr = _ConcreteASR("")
    conn.asr = asr
    answered = []

    # Production shape: audio present, so the reset branch runs and
    # handle_voice_stop is invoked. That call holds the fragment (tier-1
    # trailing token 個) — i.e. it creates the hold, it does not consume one.
    conn.asr_audio = [b"\x00" * 320]

    with mock.patch.object(asr, "_transcript", "我想話加個"):
        asr._transcript = "我想話加個"
        answered = _run_listen_stop_real_reset(conn)

    assert len(answered) == 1, (
        "the fragment held DURING handle_voice_stop was never answered — "
        f"user gets no reply (still held={conn._pending_utterance!r}): {answered}"
    )
    parsed = json.loads(answered[0])
    assert parsed["content"] == "我想話加個"
    assert conn._pending_utterance is None, "the hold must be consumed, not left"


def test_fragment_held_during_voice_stop_is_answered_on_the_stream_path():
    """Same hole on the STREAM branch: no `handle_voice_stop` runs, but the
    flush must still fire (the stop request is fire-and-forget and answers
    nothing)."""
    conn = _RealResetConn(llm=_ScriptedLLM(['{"complete": true}']))
    asr = _ConcreteASR("")
    asr.interface_type = InterfaceType.STREAM
    asr._send_stop_request = mock.AsyncMock()
    conn.asr = asr

    # Held by a previous fragment, still unanswered.
    conn._pending_utterance = "我想話加個"
    conn._pending_utterance_speaker = "Vincent"
    conn.asr_audio = []

    answered = _run_listen_stop_real_reset(conn)

    assert asr._send_stop_request.called, "stream branch must still send stop"
    assert len(answered) == 1, f"stream path dropped the hold: {answered}"
    parsed = json.loads(answered[0])
    assert parsed["content"] == "我想話加個"
    assert parsed["speaker"] == "Vincent"


def test_held_fragment_and_terminal_fragment_merge_into_ONE_answer():
    """THE de-merge regression (build-tester round 3, v0.9.14).

    The gate's whole purpose is to turn "one thought, N VAD fragments" into ONE
    answer. So when a fragment A was already held and the terminal fragment B
    arrives in this same stop pass, A and B must reach the LLM as a SINGLE
    merged utterance.

    ⚠️ The first version of the fix flushed A before `handle_voice_stop()`
    (to survive `reset_audio_states()`). That POPPED A out of the gate, so
    `handle_voice_stop` never saw it and answered B on its own — two answers
    for one thought, i.e. the original "breaks my questions into batches"
    symptom reintroduced inside the fix.

    The correct shape: capture A before the reset, RESTORE it into the gate
    after the reset, then let the single post-ASR flush answer A+B together.
    """
    conn = _RealResetConn(llm=_ScriptedLLM(['{"complete": true}']))
    asr = _ConcreteASR("")
    conn.asr = asr

    # A: an earlier thought fragment, already held by the gate.
    conn._pending_utterance = "我想問一個關於頭痛嘅"
    conn._pending_utterance_speaker = "Vincent"
    # B: the terminal fragment arriving now. Ends on 所以 → tier-1 incomplete,
    # so the gate HOLDS it rather than answering — this is what makes the
    # stop-path flush necessary.
    asr._transcript = "問題，睇下有咩方案，所以"
    conn.asr_audio = [b"\x00" * 320]

    answered = _run_listen_stop_real_reset(conn)

    assert len(answered) == 1, (
        f"one thought produced {len(answered)} answers — the held fragment A "
        f"and the terminal fragment B were NOT merged (de-merge regression): "
        f"{answered}"
    )
    parsed = json.loads(answered[0])
    content = parsed["content"]
    assert "頭痛" in content, f"held fragment A was dropped: {content!r}"
    assert "方案" in content, f"terminal fragment B was missing: {content!r}"
    assert content.index("頭痛") < content.index("方案"), "order not preserved"
    assert conn._pending_utterance is None


def test_exactly_one_answer_across_every_stop_scenario():
    """No double-answer, no dropped answer — the whole matrix.

    The double-flush shape this replaced could answer one thought twice. This
    pins the invariant for every state the handler can be in: a turn must
    produce EXACTLY ONE answer (or zero when there is genuinely nothing to
    answer), and must never leave a fragment held behind it.
    """
    cases = [
        # (name, pre-held A, terminal transcript, audio, expect_answers)
        ("A held + B incomplete -> merged", "我想問頭痛嘅", "問題，睇下有咩方案，所以", [b"\x00" * 320], 1),
        ("A held + B complete -> merged", "我想問頭痛嘅", "講翻今日新聞啦。", [b"\x00" * 320], 1),
        ("A held + no audio -> A alone", "我想話加個", "", [], 1),
        ("no hold + B incomplete -> B held then flushed", None, "我想話加個", [b"\x00" * 320], 1),
        ("no hold + B complete -> answered by gate", None, "講翻今日新聞啦。", [b"\x00" * 320], 1),
        ("nothing at all -> no turn", None, "", [], 0),
    ]
    for name, prehold, transcript, audio, expected in cases:
        conn = _RealResetConn(llm=_ScriptedLLM(['{"complete": true}']))
        asr = _ConcreteASR("")
        conn.asr = asr
        if prehold:
            conn._pending_utterance = prehold
            conn._pending_utterance_speaker = "Vincent"
        asr._transcript = transcript
        conn.asr_audio = audio

        answered = _run_listen_stop_real_reset(conn)

        assert len(answered) == expected, (
            f"[{name}] expected {expected} answer(s), got {len(answered)}: "
            f"{answered}"
        )
        assert conn._pending_utterance is None, (
            f"[{name}] left a fragment held — it can leak into the next turn"
        )
