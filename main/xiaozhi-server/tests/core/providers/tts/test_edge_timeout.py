"""RED: EdgeTTS must bound its per-attempt network timeouts.

`edge.py` was the only TTS provider ignoring `tts_timeout`. Every other provider
threads it into its HTTP client; EdgeTTS passed nothing to `edge_tts.Communicate`,
so the library defaults applied:

    connect_timeout = 10
    receive_timeout = 60   <-- a hung upstream blocks 60s PER ATTEMPT

With 2 attempts that is ~120s of dead air with no terminal signal — worse than
the 24s no-backoff bug we just fixed, because the client sits on its spinner for
two minutes.

The fix threads `self.tts_timeout` into `Communicate` so a hung upstream is
bounded by config instead of by a library default nobody chose.
"""

import inspect
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.providers.tts.edge import TTSProvider  # noqa: E402


class _Config(dict):
    def get(self, key, default=None):
        return dict.get(self, key, default)


def _make_provider(tts_timeout=15):
    cfg = _Config(
        {
            "type": "edge",
            "voice": "zh-HK-WanLungNeural",
            "output_dir": "tmp/",
            "tts_timeout": tts_timeout,
        }
    )
    # bypass network-touching __init__ work; we only want the config-derived state
    prov = TTSProvider.__new__(TTSProvider)
    prov.tts_timeout = tts_timeout
    prov.edge_rate = "+0%"
    prov.edge_volume = "+0%"
    prov.edge_pitch = "+0Hz"
    prov.voice = "zh-HK-WanLungNeural"
    prov._lang_voices = {}
    return prov


def test_edge_text_to_speak_passes_timeouts_to_communicate(monkeypatch):
    """`Communicate` must receive explicit timeouts derived from tts_timeout."""
    captured = {}

    class _FakeCommunicate:
        def __init__(self, text, **kwargs):
            captured.update(kwargs)

        async def stream(self):
            if False:  # pragma: no cover - empty async generator
                yield b""

    import core.providers.tts.edge as edge_mod

    monkeypatch.setattr(edge_mod.edge_tts, "Communicate", _FakeCommunicate)

    prov = _make_provider(tts_timeout=15)

    import asyncio

    asyncio.run(prov.text_to_speak("你好", None))

    assert "connect_timeout" in captured, (
        "EdgeTTS passed no connect_timeout to edge_tts.Communicate — the library "
        "default (10s) applies, so tts_timeout is ignored."
    )
    assert "receive_timeout" in captured, (
        "EdgeTTS passed no receive_timeout to edge_tts.Communicate — the library "
        f"default (60s) applies. A hung upstream blocks 60s PER ATTEMPT (~120s "
        "for 2 attempts) with no terminal signal. THIS IS THE BUG."
    )


def test_edge_receive_timeout_derives_from_tts_timeout(monkeypatch):
    """The values must come from tts_timeout, not be hardcoded library defaults."""
    captured = {}

    class _FakeCommunicate:
        def __init__(self, text, **kwargs):
            captured.update(kwargs)

        async def stream(self):
            if False:  # pragma: no cover
                yield b""

    import core.providers.tts.edge as edge_mod

    monkeypatch.setattr(edge_mod.edge_tts, "Communicate", _FakeCommunicate)

    prov = _make_provider(tts_timeout=7)

    import asyncio

    asyncio.run(prov.text_to_speak("你好", None))

    assert captured.get("receive_timeout") == 7, (
        f"receive_timeout should be tts_timeout (7), got "
        f"{captured.get('receive_timeout')!r} — hardcoded default means config is ignored"
    )


def test_edge_timeout_bounds_are_positive_and_finite():
    """A misconfigured tts_timeout must never disable the bound."""
    prov = _make_provider(tts_timeout=15)

    import core.providers.tts.edge as edge_mod

    src = inspect.getsource(edge_mod.TTSProvider.text_to_speak)
    assert "receive_timeout" in src, "text_to_speak does not reference receive_timeout"
    assert "self.tts_timeout" in src, (
        "text_to_speak does not reference self.tts_timeout — the bound is not "
        "config-derived"
    )
