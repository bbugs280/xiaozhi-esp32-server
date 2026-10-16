"""The TTS segmenter must strip markdown before handing text to the synthesiser.

⚠️ WHY THIS FILE EXISTS (2026-10-16):
Vincent's articulation coach replies contain markdown (`**Week 3 Challenge**`,
`*"What breaks first?"*`). The TTS segmenter only trims punctuation/emoji from
the ENDS of a segment and never removes markdown markers, so the synthesiser
receives them literally. The server log shows it:

    语音生成成功: **Week 3 Challenge — Impromptu 2-Minute, No Prep**

Markdown syntax is a DISPLAY format. It exists for the screen, not for the ear,
and a voice assistant reading "asterisk asterisk" aloud is a defect regardless
of whether a given engine happens to tolerate it.

This strips the markers only — never the words. Content preservation is an
explicit invariant below, because a TTS text cleaner that drops content is far
worse than one that reads a stray asterisk.
"""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.utils import textUtils as T  # noqa: E402


def _clean(text):
    fn = getattr(T, "strip_markdown_for_speech", None)
    assert fn is not None, (
        "textUtils has no strip_markdown_for_speech() \u2014 the TTS pipeline "
        "passes `**bold**` markers to the synthesiser verbatim"
    )
    return fn(text)


def test_bold_markers_are_removed():
    assert _clean("**Week 3 Challenge**") == "Week 3 Challenge"


def test_italic_markers_are_removed():
    assert _clean('*"What breaks first?"*') == '"What breaks first?"'


def test_underscore_emphasis_removed():
    assert _clean("__Part A__") == "Part A"


def test_headings_markers_removed():
    assert _clean("## Part A") == "Part A"
    assert _clean("# Heading") == "Heading"


def test_list_markers_removed():
    assert _clean("- first item") == "first item"
    assert _clean("* bullet point") == "bullet point"


def test_inline_code_backticks_removed():
    assert _clean("run `npm test` now") == "run npm test now"


def test_word_content_is_never_lost():
    """THE INVARIANT: cleaning must not delete the words.

    A TTS cleaner that eats content is worse than one that reads an asterisk.
    """
    text = (
        "**Part A — 60 seconds of silence (3 min)**\n"
        "No talking, no notes. Decide your container."
    )
    out = _clean(text)
    for word in ["Part A", "60", "seconds", "silence", "No talking",
                 "notes", "container"]:
        assert word in out, f"content lost: {word!r} missing from {out!r}"
    assert "**" not in out


def test_plain_prose_is_untouched():
    text = "今日天氣凍，帶件外套。"
    assert _clean(text) == text


def test_underscores_inside_words_are_preserved():
    """`snake_case_names` must not be mangled into `snakecasenames`."""
    assert _clean("call snake_case_name here") == "call snake_case_name here"


def test_link_becomes_readable_text():
    """A markdown link should speak its label, not the URL."""
    out = _clean("see [the docs](https://example.com/x)")
    assert "the docs" in out
    assert "https://" not in out


def test_the_real_coach_reply_is_clean_but_complete():
    """End-to-end shape check against the actual reported content."""
    reply = (
        '**Week 3 Challenge — Impromptu 2-Minute, No Prep**\n\n'
        "Your topic is not yours to choose. "
        'Last session\'s draw was set and never run, so we keep it: '
        '*"What breaks first when a team doubles in size?"*\n'
        "No re-draw, no reading ahead.\n\n"
        "**Part A — 60 seconds of silence (3 min)**\n"
        "No talking, no notes.\n"
    )
    out = _clean(reply)

    assert "**" not in out, "bold markers still present \u2014 would be spoken aloud"
    for must_keep in [
        "Week 3 Challenge",
        "Your topic is not yours to choose",
        "What breaks first when a team doubles in size",
        "Part A",
        "60 seconds of silence",
    ]:
        assert must_keep in out, f"content lost: {must_keep!r}"


def test_whitespace_and_line_structure_preserved():
    """Stripping markdown must NOT collapse the line structure.

    The iOS renderer needs the newlines to lay the reply out; this function
    must not become a second source of the dense-block bug.
    """
    reply = "**A**\n\nSome text.\n\n**B**\nMore text."
    out = _clean(reply)
    assert "\n\n" in out, f"paragraph break lost: {out!r}"
    assert out.count("\n") >= 3, f"line breaks lost: {out!r}"


# --------------------------------------------------------------------------
# Build-tester findings (2026-10-16) — both were REAL bugs found by attacking
# this function, and both are regression-guarded here.
# --------------------------------------------------------------------------


def test_emphasis_removal_does_not_glue_words_together():
    """⚠️ BUILD-TESTER FINDING: `Try *saying*it slower.` became
    `Try sayingit slower` — removing the markers joined two words, producing
    a mispronounced word. That is WORSE than speaking a stray asterisk, so it
    is a content bug, not a cosmetic one.
    """
    out = _clean("Try *saying*it slower.")
    assert "sayingit" not in out, (
        f"emphasis removal glued two words into a non-word: {out!r}"
    )
    assert "saying" in out and "it" in out


def test_emphasis_glue_guard_leaves_normal_cases_intact():
    """The glue guard must not put spaces where none are wanted."""
    assert _clean("*word*") == "word"
    assert _clean('*"quoted text"*') == '"quoted text"'
    assert _clean("normal *emphasis* here") == "normal emphasis here"


def test_link_regex_is_linear_not_quadratic():
    """⚠️ BUILD-TESTER FINDING: the link pattern was O(n²) on unmatched
    brackets — 4.1s for 50KB. The TTS segment path is a hot path, so a
    pathological model output could stall speech generation.
    """
    import time

    pathological = "[" * 20000
    t0 = time.perf_counter()
    _clean(pathological)
    dt = time.perf_counter() - t0
    assert dt < 0.5, (
        f"stripper took {dt:.2f}s on 20k unmatched brackets — the link regex "
        f"is quadratic and can stall the TTS hot path"
    )


def test_link_still_works_after_linear_rewrite():
    assert _clean("see [the docs](https://example.com/x)") == "see the docs"
    assert _clean("[a](b) [c](d)") == "a c"
