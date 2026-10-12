import time
import uuid
import asyncio
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

from core.utils.dialogue import Message
from core.providers.asr.dto.dto import InterfaceType
from core.handle.receiveAudioHandle import startToChat
from core.handle.reportHandle import enqueue_asr_report
from core.handle.sendAudioHandle import send_stt_message, send_tts_message
from core.handle.textMessageHandler import TextMessageHandler
from core.handle.textMessageType import TextMessageType
from core.utils.util import remove_punctuation_and_length
from core.providers.tts.dto.dto import ContentType, TTSMessageDTO, SentenceType


TAG = __name__

class ListenTextMessageHandler(TextMessageHandler):
    """Listen消息处理器"""

    @property
    def message_type(self) -> TextMessageType:
        return TextMessageType.LISTEN

    async def handle(self, conn: "ConnectionHandler", msg_json: Dict[str, Any]) -> None:
        if "mode" in msg_json:
            conn.client_listen_mode = msg_json["mode"]
            conn.logger.bind(tag=TAG).debug(
                f"客户端拾音模式：{conn.client_listen_mode}"
            )
        if msg_json["state"] == "start":
            # 设备从播放模式切回录音模式,清除所有音频状态和缓冲区
            conn.reset_audio_states()
            # ⚠️ Clear the abort flag too. An earlier STOP/barge-in set
            # client_abort=True, and reset_audio_states() only clears audio/VAD
            # state — the flag itself stays latched True into THIS turn. That
            # makes the next spoken utterance hit the `startToChat` guard
            # ("client_abort set during intent routing — dropping the turn") and
            # be silently discarded — the user's "after I stop to interrupt it
            # takes a while to resume" bug (the first re-speak is dropped, so
            # they must speak twice). A fresh `listen start` is exactly the
            # boundary where a new turn begins, so this is where the flag must
            # reset.
            conn.client_abort = False
        elif msg_json["state"] == "stop":
            # 收到stop但asr未初始化，跳过处理
            if conn.asr is None:
                return

            conn.client_voice_stop = True

            # ⚠️ `listen stop` is FINAL — no further audio will arrive, so every
            # fragment held by the time this handler returns must be answered.
            #
            # Two things fight here:
            #   • a hold carried in from an EARLIER fragment, which
            #     `reset_audio_states()` below clears (so it must be saved
            #     BEFORE that reset, or it vanishes);
            #   • the gate's job of MERGING that earlier fragment with the
            #     terminal one arriving now, so one thought yields ONE answer.
            #
            # Flushing the earlier fragment here would satisfy the first and
            # defeat the second — it pops A out of the gate, `handle_voice_stop`
            # never sees it, and A and B are answered as two turns. That is the
            # original "breaks my questions into batches" symptom rebuilt inside
            # the fix.
            #
            # So: SAVE A across the reset, RESTORE it, let the ASR call merge
            # A+B, then flush once at the end. Single answer, everything
            # preserved.
            # ⚠️ Read the speaker BEFORE taking the text: `take_pending_utterance()`
            # clears BOTH (they are one unit), so reading it afterwards always
            # yields None and silently drops attribution.
            saved_speaker = conn.take_pending_utterance_speaker()
            saved_raw = conn.take_pending_utterance()
            asr_ran = False

            if conn.asr.interface_type == InterfaceType.STREAM:
                # 流式模式下，发送结束请求
                asyncio.create_task(conn.asr._send_stop_request())
            else:
                # 非流式模式：直接触发ASR识别
                if len(conn.asr_audio) > 0:
                    asr_audio_task = conn.asr_audio.copy()
                    conn.reset_audio_states()

                    # ⚠️ Restore A HERE — after the reset (which cleared it) and
                    # BEFORE `handle_voice_stop()`. The gate reads
                    # `_pending_utterance` to merge A with the terminal fragment
                    # B; restoring after that call is too late, because by then
                    # the gate has already held B alone and A is stranded.
                    if saved_raw:
                        conn._pending_utterance = saved_raw
                        conn._pending_utterance_speaker = saved_speaker

                    if len(asr_audio_task) > 0:
                        asr_ran = True
                        await conn.asr.handle_voice_stop(conn, asr_audio_task)

            # If `handle_voice_stop` ran, it consumed A into its own decision
            # (merged-and-answered, or merged-and-re-held). Anything still held
            # is answered by the single flush below. When it did NOT run —
            # the stream branch, or no audio — A is still unanswered and must
            # survive to that flush.
            if saved_raw and not asr_ran and conn._pending_utterance is None:
                conn._pending_utterance = saved_raw
                conn._pending_utterance_speaker = saved_speaker

            # ── Answer whatever the gate is holding ────────────────────────
            # A fragment is only held because it looked incomplete; with the
            # user done talking, that text is all we will ever get. The payload
            # from `flush_pending_utterance()` is already re-wrapped (voiceprint
            # JSON when a speaker was identified) so it reaches the LLM in the
            # same shape as a normal turn.
            held_payload = conn.flush_pending_utterance()
            if held_payload:
                conn.logger.bind(tag=TAG).info(
                    f"listen stop -> flushing held fragment: {held_payload}"
                )
                await startToChat(conn, held_payload)
        elif msg_json["state"] == "detect":
            conn.client_have_voice = False
            conn.reset_audio_states()
            if "text" in msg_json:
                conn.last_activity_time = time.time() * 1000
                original_text = msg_json["text"]  # 保留原始文本
                filtered_len, filtered_text = remove_punctuation_and_length(
                    original_text
                )

                # 检查是否是设备呼叫指令 [device_call]
                if original_text.startswith("[device_call]"):
                    # 提取 tag 后的文本
                    call_text = original_text[len("[device_call]"):].strip()
                    conn.logger.bind(tag=TAG).info(f"收到设备呼叫指令: {call_text}")

                    # 标记为来电接听模式
                    conn.incoming_call = True

                    # 准备开始新会话
                    conn.sentence_id = uuid.uuid4().hex

                    await send_stt_message(conn, call_text)

                    # 等待tts初始化，最多等待3秒
                    start_time = time.time()
                    while time.time() - start_time < 3:
                        if conn.tts:
                            break
                        await asyncio.sleep(0.1)

                    if conn.tts:
                        conn.tts.store_tts_text(conn.sentence_id, call_text)
                        conn.tts.tts_text_queue.put(TTSMessageDTO(sentence_id=conn.sentence_id, sentence_type=SentenceType.FIRST, content_type=ContentType.ACTION))
                        conn.tts.tts_one_sentence(conn, ContentType.TEXT, content_detail=call_text)
                        conn.tts.tts_text_queue.put(TTSMessageDTO(sentence_id=conn.sentence_id, sentence_type=SentenceType.LAST, content_type=ContentType.ACTION))

                    # 添加到对话历史，让模型理解上下文
                    conn.dialogue.put(Message(role="assistant", content=call_text))
                    return

                # 识别是否是唤醒词
                is_wakeup_words = filtered_text in conn.config.get("wakeup_words")
                # 是否开启唤醒词回复
                enable_greeting = conn.config.get("enable_greeting", True)

                if is_wakeup_words and not enable_greeting:
                    # 如果是唤醒词，且关闭了唤醒词回复，就不用回答
                    await send_stt_message(conn, original_text)
                    await send_tts_message(conn, "stop", None)
                    conn.client_is_speaking = False
                elif is_wakeup_words:
                    conn.just_woken_up = True
                    # 上报纯文字数据（复用ASR上报功能，但不提供音频数据）
                    enqueue_asr_report(conn, "嘿，你好呀", [])
                    await startToChat(conn, "嘿，你好呀")
                else:
                    conn.just_woken_up = True
                    # 上报纯文字数据（复用ASR上报功能，但不提供音频数据）
                    enqueue_asr_report(conn, original_text, [])
                    # 否则需要LLM对文字内容进行答复
                    await startToChat(conn, original_text)