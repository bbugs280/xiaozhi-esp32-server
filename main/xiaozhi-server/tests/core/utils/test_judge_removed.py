"""RED tests for REMOVING the LLM completeness judge (option (a), 2026-10-16).

## Why the judge is being deleted

Built 2026-10-11 to stop the LLM answering half-sentences. Measured against the
FULL production log (~772 hold events, `tmp/server.log`), it is a net negative:

**1. It almost never sees real speech.** Of 772 hold events, 768 are test
fixtures — the same 5 canned sentences replayed by the test suite. Only 4 holds
came from Vincent actually speaking.

**2. On real speech its verdicts were wrong.** Both real holds from 2026-10-16
were COMPLETE utterances that it wrongly held:

    11:33:06  可唔可以啊？    -> held, but this is a complete question
    11:33:37  電下講落去。    -> held, but this is a complete command

Each misfire is seconds of silence in which the user believes they were not
heard — the mechanism behind "I have to say it 3 times".

**3. Its one consequential success still hurt.** `天氣啊。` was held ~3s
(11:39:51), the user repeated themselves, and the turn was then DROPPED
(`client_abort set during intent routing`). Holding caused the symptom rather
than preventing it.

**4. Tier 1 never fires.** 0 `tier1_incomplete` events in the entire log. The
"zero added latency in the common case" claim was false; every utterance paid
a network round-trip.

**5. There is no evidence Vincent ever pauses mid-thought.** The merged-fragment
case the judge was built for (`我想問一個關於頭痛嘅` + `問題啦，因為…`)
appears ONLY in test fixtures, never in real speech.

## What replaces it

Nothing. `is_utterance_complete` is deleted, and the ASR path answers every
utterance immediately — the pre-gate behaviour. This removes:
  - ~0.6s of latency per utterance (the tier-2 budget tax)
  - the misfire class (wrongly-held complete questions)
  - the merge/hold machinery and its own failure modes

⚠️ These tests are RED while the judge still exists.
"""

import inspect

import pytest


def test_judge_module_is_deleted():
    """The LLM judge module must be gone, not merely unused.

    Leaving dead code behind is how the completeness judge itself stayed
    broken-but-present for a month. If the decision is to remove it, remove it.
    """
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("core.utils.utterance_completeness")


def test_no_llm_judge_function_remains():
    """`judge_completeness_llm` and its helpers must not survive the deletion.

    RED: all four still exist today.
    """
    import importlib

    try:
        uc = importlib.import_module("core.utils.utterance_completeness")
    except ModuleNotFoundError:
        return  # deleted: the goal state

    for name in (
        "judge_completeness_llm",
        "_consume_llm_result",
        "_parse_complete",
        "_JUDGE_BUDGET_S",
        "_JUDGE_TIMEOUT_S",
    ):
        assert not hasattr(uc, name), (
            f"{name} still exists — the LLM judge was not actually removed"
        )


def test_asr_path_no_longer_imports_the_gate():
    """The ASR provider must not reference the gate at all.

    A dangling import or call is the latent-AttributeError failure mode: the
    module is deleted, the import is not, and every turn raises inside a broad
    `except` — silently restoring the bug. Assert on the real source.
    """
    import core.providers.asr.base as base

    src = inspect.getsource(base)
    for forbidden in (
        "is_utterance_complete",
        "utterance_completeness",
        "_pending_utterance",
        "flush_pending_utterance",
    ):
        assert forbidden not in src, (
            f"core/providers/asr/base.py still references {forbidden!r} — "
            "a deleted module left a live call site"
        )


def test_no_module_in_repo_still_imports_the_gate():
    """Nothing anywhere may import the deleted module.

    Scans the real tree rather than a hand-written list, so a forgotten call
    site in a file we did not think of still fails the test.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3]  # main/xiaozhi-server
    offenders = []
    for py in root.rglob("*.py"):
        # Skip the tests themselves: they legitimately name the module.
        # Also skip scratch dirs (tmp/) — probe scripts are not production
        # code sites and would fail this scan for the wrong reason.
        if "tests/" in str(py) or "tmp/" in str(py) or not py.is_file():
            continue
        if "__pycache__" in str(py):
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "utterance_completeness" in text:
            offenders.append(str(py.relative_to(root)))

    assert not offenders, (
        f"these files still import the deleted judge: {offenders}"
    )
