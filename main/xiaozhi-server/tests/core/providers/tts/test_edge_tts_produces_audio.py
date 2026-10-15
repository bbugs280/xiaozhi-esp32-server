"""Regression test: the Edge-TTS provider must actually produce audio.

Bug (2026-10-15, present through v0.9.14): every Edge-TTS synthesis call failed
with `WSServerHandshakeError: 403, message='Invalid response status'` against
`wss://speech.platform.bing.com/.../edge/v1`.

Root cause: Microsoft bumped the `Sec-MS-GEC-Version` scheme from `1-<ver>` to
`2-<ver>`. The pinned `edge_tts==7.2.6` (and the installed 7.2.8) both send
`1-143.0.3650.75` and are rejected. `edge-tss>=7.3.0` sends `2-143.0.3650.75`.

User-visible symptom (Vincent's words): "it's still stuck at thinking…" and
"it doesn't seem to be responding". Server-side the log showed, for every
sentence of every reply:

    语音生成失败1..5次: <text>，错误: Edge TTS请求失败: 403, ...
    发送第一段语音: None

i.e. the LLM answered correctly, then ALL retries failed and the server pushed
`None` audio — so the client received no audio frame and no error frame, and
sat on its spinner forever.

This test asserts the SYMPTOM, not an adjacent invariant: does a call through
the project's real TTS provider entry point return actual audio bytes?

⚠️ This test makes a live network call to Microsoft's Edge-TTS endpoint. Only a
genuine network outage is skipped; an HTTP 403 is a FAILURE, not a skip —
403 is the exact bug this guards, and treating it as "unreachable" would make
the test vacuous (green while the user's replies are silent).
"""

import asyncio

import pytest

from core.providers.tts.edge import TTSProvider


def _make_provider():
    """Build a TTSProvider with the minimum config the server passes in prod.

    Mirrors `_initialize_tts`: the hello message injects a per-language voice
    map, which takes precedence over the config default.
    """
    return TTSProvider(
        config={"voice": {"yue": "zh-HK-HiuMaanNeural", "en": "en-HK-YanNeural"}},
        delete_audio_file=True,
    )


def test_edge_provider_returns_audio_bytes():
    """The provider must return non-empty audio for a normal Cantonese sentence."""
    provider = _make_provider()

    try:
        audio = asyncio.run(provider.text_to_speak("今日天氣唔錯。", None))
    except Exception as exc:
        if "403" in str(exc):
            # NOT a skip — a 403 is the exact defect this test guards.
            pytest.fail(f"Edge-TTS rejected the request (the reported bug): {exc}")
        if "Timeout" in str(exc) or "Cannot connect" in str(exc):
            pytest.skip(f"Edge-TTS endpoint unreachable (network outage): {exc}")
        raise

    # The symptom Vincent reported: no audio comes back.
    assert audio, "Edge-TTS returned no audio bytes — the reply will be silent"
    assert len(audio) > 1000, f"suspiciously small audio payload: {len(audio)} bytes"


def test_edge_provider_returns_audio_for_english_too():
    """Pin the opposite direction: the English voice path must also produce audio.

    A fix that only unblocks the Cantonese voice would still leave English
    replies silent — and `_resolve_voice` picks the voice from the reply text,
    so both branches need to work.
    """
    provider = _make_provider()

    try:
        audio = asyncio.run(provider.text_to_speak("The weather is fine today.", None))
    except Exception as exc:
        if "403" in str(exc):
            pytest.fail(f"Edge-TTS rejected the request (the reported bug): {exc}")
        if "Timeout" in str(exc) or "Cannot connect" in str(exc):
            pytest.skip(f"Edge-TTS endpoint unreachable (network outage): {exc}")
        raise

    assert audio, "Edge-TTS returned no audio bytes for an English reply"
    assert len(audio) > 1000, f"suspiciously small audio payload: {len(audio)} bytes"
