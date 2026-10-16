"""The TTS segmenter must ACTUALLY APPLY the markdown stripper's result.

⚠️ WHY THIS FILE EXISTS:
A cleaner function nobody calls fixes nothing — the same "test the wiring, not
just the function" mistake build-tester caught three times in this project (the
gate short-circuit, the embedder prewarm, and now this).

⚠️ REWRITTEN after build-tester found this version was only a token-presence
proxy. The first version did `ast.unparse(node)` + substring check, and FOUR
realistic evasions stayed green — including the worst one, E1:

    _cleaned = textUtils.strip_markdown_for_speech(segment_text_raw)
    segment_text = textUtils.get_string_no_punctuation_or_emoji(segment_text_raw)
                                                                ^^^^^^^^^^^^^^^
                                              original, result DISCARDED

That is "wired but not applied": the call exists, the name appears in the AST,
the synthesiser still receives `**Week 3 Challenge**`. It is precisely the bug
class this file's docstring claims to prevent.

The fix: assert the stripper's result is what gets PASSED ON — i.e. its call
node is nested INSIDE the punctuation-trimming call, not a sibling statement.
"""

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TARGET = PROJECT_ROOT / "core" / "providers" / "tts" / "base.py"
STRIPPER = "strip_markdown_for_speech"
WRAPPER = "get_string_no_punctuation_or_emoji"


def _tree():
    return ast.parse(TARGET.read_text())


def _applied_in(fn_name):
    """True if, inside `fn_name`, the stripper's result is FEED INTO the
    punctuation-trimming call (so it actually reaches the synthesiser).

    A bare call whose value is thrown away returns False.
    """
    for node in ast.walk(_tree()):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != fn_name:
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            fname = getattr(call.func, "attr", None) or getattr(call.func, "id", None)
            if fname != WRAPPER:
                continue
            # is the stripper invoked anywhere INSIDE this wrapper's args?
            for arg in call.args:
                for inner in ast.walk(arg):
                    if isinstance(inner, ast.Call):
                        iname = (
                            getattr(inner.func, "attr", None)
                            or getattr(inner.func, "id", None)
                        )
                        if iname == STRIPPER:
                            return True
        return False
    return False


def test_segmenter_applies_markdown_stripper():
    """`_get_segment_text` must feed the stripper's OUTPUT into the trimmer."""
    assert _applied_in("_get_segment_text"), (
        "_get_segment_text does not APPLY strip_markdown_for_speech() — either "
        "it is missing, or (the E1 evasion) it is called and its result "
        "discarded, so `**bold**` markers still reach the synthesiser"
    )


def test_remaining_text_path_applies_markdown_stripper():
    """The tail path must also feed the stripper's output onward.

    The tail carries whatever is left after the final punctuation, which for a
    structured reply is often the whole last section.
    """
    assert _applied_in("_process_remaining_text_stream"), (
        "_process_remaining_text_stream does not APPLY the stripper — the tail "
        "of a structured reply would still be spoken with literal markers"
    )


def test_target_file_exists():
    """Loud failure if base.py moves, rather than a silent pass."""
    assert TARGET.exists(), f"expected TTS segmenter at {TARGET}"
