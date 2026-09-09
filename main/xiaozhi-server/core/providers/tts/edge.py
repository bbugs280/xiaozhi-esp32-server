import os
import uuid
import edge_tts
from datetime import datetime
from core.providers.tts.base import TTSProviderBase


class TTSProvider(TTSProviderBase):
    TTS_PARAM_CONFIG = [
        ("ttsVolume", "volume", 0, 100, 50, int),
        ("ttsRate", "speech_rate", -100, 100, 0, int),
        ("ttsPitch", "pitch_rate", -100, 100, 0, int),
    ]

    # Voice per reply language (mirrors connection.py _detect_language convention).
    # Edge-TTS reads every glyph through the SELECTED voice's reading system, so a
    # Cantonese voice (zh-HK-*) will mangle ASCII digits/English loanwords in an
    # English reply. Key the voice off the reply text's own CJK presence, so a
    # Cantonese reply uses the Cantonese voice and an English reply uses an
    # English voice — no segmentation/concatenation needed.
    LANG_VOICES = {
        "yue": "zh-HK-WanLungNeural",
        "en": "en-HK-SamNeural",
    }

    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)

        # "voice" may be a single string (default) or a per-language mapping.
        voice_cfg = config.get("voice")
        self._lang_voices = None
        if isinstance(voice_cfg, dict):
            # config overrides the default per-language map
            self._lang_voices = voice_cfg
            self.voice = None
        elif isinstance(voice_cfg, str) and voice_cfg:
            self.voice = voice_cfg
            self._lang_voices = None
        else:
            # no voice configured — fall back to per-language defaults
            self.voice = None
            self._lang_voices = self.LANG_VOICES

        if config.get("private_voice"):
            self.voice = config.get("private_voice")
            self._lang_voices = None
        self.audio_file_type = config.get("format", "mp3")

        volume = config.get("volume", "50")
        self.volume = int(volume) if volume else 50

        speech_rate = config.get("rate", "0")
        self.speech_rate = int(speech_rate) if speech_rate else 0

        pitch_rate = config.get("pitch", "0")
        self.pitch_rate = int(pitch_rate) if pitch_rate else 0

        # 应用百分比调整
        self._apply_percentage_params(config)

        self.edge_rate = f"{self.speech_rate:+}%"
        self.edge_volume = f"{self.volume:+}%"
        self.edge_pitch = f"{self.pitch_rate:+}Hz"

    def set_client_voice(self, voice_map: dict) -> None:
        """Inject a per-connection voice override ({"yue": ..., "en": ...}).

        Called by `ConnectionHandler._initialize_tts` after the hello message
        carries a `voice` field. Takes precedence over `data/.config.yaml`'s
        default per-language map for this connection only.
        """
        if isinstance(voice_map, dict):
            self._lang_voices = voice_map
            self.voice = None

    def generate_filename(self, extension=".mp3"):
        return os.path.join(
            self.output_file,
            f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}",
        )

    def _resolve_voice(self, text: str) -> str:
        """Pick the Edge-TTS voice for a reply based on its own language.

        If `self.voice` is set (explicit single voice or private_voice), use it
        verbatim — preserves the old fixed-voice behaviour. Otherwise, if a
        per-language map is configured, detect the reply's language by CJK
        presence and return the matching voice (English default).
        """
        if self.voice:
            return self.voice
        if self._lang_voices:
            lang = "yue" if any("\u4e00" <= ch <= "\u9fff" for ch in (text or "")) else "en"
            return self._lang_voices.get(lang) or self._lang_voices.get("en") or ""
        return self.voice

    async def text_to_speak(self, text, output_file):
        try:
            voice = self._resolve_voice(text)
            communicate = edge_tts.Communicate(
                text,
                voice=voice,
                rate=self.edge_rate,
                volume=self.edge_volume,
                pitch=self.edge_pitch,
            )
            if output_file:
                # 确保目录存在并创建空文件
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "wb") as f:
                    pass

                # 流式写入音频数据
                with open(output_file, "ab") as f:  # 改为追加模式避免覆盖
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":  # 只处理音频数据块
                            f.write(chunk["data"])
            else:
                # 返回音频二进制数据
                audio_bytes = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_bytes += chunk["data"]
                return audio_bytes
        except Exception as e:
            error_msg = f"Edge TTS请求失败: {e}"
            raise Exception(error_msg)  # 抛出异常，让调用方捕获