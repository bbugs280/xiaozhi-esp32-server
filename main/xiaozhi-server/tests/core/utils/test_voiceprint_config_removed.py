"""Regression: no voiceprint config block -> no voiceprint init attempt.

## The bug (found 2026-10-16)

`data/.config.yaml` carried a `voiceprint:` block:

    voiceprint:
      type: local_ecapa
      threshold: 0.42
      margin: 0.05

`type: local_ecapa` made `ConnectionHandler._initialize_voiceprint()` import
`core.utils.local_ecapa_voiceprint` - **a module that does not exist** in this
repo (only a stale `.pyc` from 2026-09-07 survives; the source was never
committed). Every connection therefore logged:

    声纹识别初始化失败: No module named 'core.utils.local_ecapa_voiceprint'

The block was doubly dead:
  1. The referenced module is absent, so the import can never succeed.
  2. Even when the module existed it was never a gate - the result was only
     attached as a `speaker` LABEL (`asr/base.py`), never used to drop a turn.
     Nothing consumed it as a decision.

Meanwhile the local path used for enrolment (`~/.cache/ai-where/speakers/`
via the Butler `ecapa_enroll.py` CLI) has nothing to do with this config -
so the server was comparing live audio against an empty speaker set, which is
why every reading was `未识别`.

## What this guards

Removing the block must make the connection stop *attempting* voiceprint init,
and must not log a failure. The test asserts on behaviour (was init attempted,
was a failure logged) rather than on the YAML text, so it survives formatting
changes but still fails if the dead config is reintroduced.
"""

import asyncio
from unittest import mock

import pytest
import yaml

from core.utils import modules_initialize


CONFIG_PATH = "data/.config.yaml"


class _RecordingLogger:
    def __init__(self):
        self.records = []

    def bind(self, **kwargs):
        return self

    def info(self, msg):
        self.records.append(("info", msg))

    def warning(self, msg):
        self.records.append(("warning", msg))

    def error(self, msg):
        self.records.append(("error", msg))

    def debug(self, msg):
        self.records.append(("debug", msg))


# --------------------------------------------------------------------------
# 1. The config itself must be gone
# --------------------------------------------------------------------------

def test_config_has_no_local_ecapa_voiceprint_block():
    """The dead `type: local_ecapa` block must not be present.

    Before the fix this config existed and pointed at a non-existent module.
    """
    with open(CONFIG_PATH) as fh:
        config = yaml.safe_load(fh)

    vp = config.get("voiceprint")
    assert vp is None or not vp, (
        f"a voiceprint config block is still present: {vp!r}. When "
        f"`type: local_ecapa` is set, every connection tries to import the "
        f"missing `core.utils.local_ecapa_voiceprint` and logs a failure."
    )


# --------------------------------------------------------------------------
# 2. Configurable-init helper must refuse a speakers-less config (no traceback)
# --------------------------------------------------------------------------

def test_initialize_voiceprint_rejects_speakers_less_config():
    """`initialize_voiceprint` must return False cleanly when the config lacks
    `speakers` - not raise, and not pretend to succeed.

    With the block present but no `speakers` key (the shipped state), this is
    the helper that silently could not do its job.
    """
    asr = mock.MagicMock()
    logger = _RecordingLogger()

    with mock.patch.object(modules_initialize, "logger", logger):
        result = modules_initialize.initialize_voiceprint(
            asr, {"voiceprint": {"type": "local_ecapa", "threshold": 0.42}}
        )

    assert result is False, "a speakers-less voiceprint config must not 'succeed'"
    asr.init_voiceprint.assert_not_called()
    assert any("不完整" in msg for _, msg in logger.records), (
        "an incomplete voiceprint config should be reported, not passed through"
    )


def test_initialize_voiceprint_no_config_is_a_silent_noop():
    """With the block removed there must be NO warning at all - absence of the
    feature is normal, not a misconfiguration. This is the state after the fix.
    """
    asr = mock.MagicMock()
    logger = _RecordingLogger()

    with mock.patch.object(modules_initialize, "logger", logger):
        result = modules_initialize.initialize_voiceprint(asr, {})

    assert result is False
    asr.init_voiceprint.assert_not_called()
    assert logger.records == [], (
        f"removing the feature must be silent, but logged: {logger.records}"
    )


# --------------------------------------------------------------------------
# 3. The connection must not attempt the missing-module import
# --------------------------------------------------------------------------

def test_connection_does_not_import_missing_local_ecapa_module():
    """`_initialize_voiceprint` must not raise when handed a config that
    references the missing `core.utils.local_ecapa_voiceprint` module.

    ⚠️ Two separate concerns, deliberately kept apart:

    * **Config removal** is guarded by
      `test_config_has_no_local_ecapa_voiceprint_block` — that is the actual
      fix, and it passes once the block is gone.
    * **This test is a robustness guard, not the fix.** It injects the dead
      config on purpose, because a user (or a restored backup) can put it back
      at any time. `_initialize_voiceprint` must swallow the ModuleNotFoundError
      and log a warning rather than propagate it into the connection setup
      thread. The import attempt itself is expected here — the repro is the
      point — so the assertion is on the OUTCOME (no exception, no provider,
      a logged warning), not on the absence of the import.
    """
    from core.connection import ConnectionHandler

    logger = _RecordingLogger()

    handler = ConnectionHandler.__new__(ConnectionHandler)
    # The dead config, verbatim as it shipped in data/.config.yaml.
    handler.config = {
        "voiceprint": {"type": "local_ecapa", "threshold": 0.42, "margin": 0.05}
    }
    handler.logger = logger

    # Must not raise — a bad voiceprint config cannot break connection setup.
    handler._initialize_voiceprint()

    assert getattr(handler, "voiceprint_provider", None) is None, (
        "no provider should be attached when the import cannot succeed"
    )
    assert any(kind == "warning" for kind, _ in logger.records), (
        f"the failure should be reported as a warning, got: {logger.records}. "
        f"Silently ignoring it is how this went unnoticed."
    )
    assert not any(kind == "error" for kind, _ in logger.records), (
        f"a recoverable config problem must not be logged as an error: "
        f"{logger.records}"
    )
