import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler
TAG = __name__


async def handleAbortMessage(conn: "ConnectionHandler"):
    conn.logger.bind(tag=TAG).info("Abort message received")
    # 设置成打断状态，会自动打断llm、tts任务
    conn.close_after_chat = False
    conn.client_abort = True
    conn.clear_queues()
    # ⚠️ Flush ASR/VAD audio buffers too. clear_queues() only drains
    # TTS/report queues; without reset_audio_states() the half-buffered
    # utterance re-fires handle_voice_stop → startToChat after the abort, and
    # (because the earlier drop already cleared client_abort) the interrupted
    # message is answered next round. reset_audio_states() clears asr_audio and
    # the VAD stop-trigger state so nothing re-fires with stale audio.
    conn.reset_audio_states()
    # 打断客户端说话状态
    await conn.websocket.send(
        json.dumps({"type": "tts", "state": "stop", "session_id": conn.session_id})
    )
    conn.clearSpeakStatus()
    conn.logger.bind(tag=TAG).info("Abort message received-end")
