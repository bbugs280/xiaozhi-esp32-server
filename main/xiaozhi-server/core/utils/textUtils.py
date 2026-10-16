import re
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
EMOJI_MAP = {
    "😂": "funny",
    "😭": "crying",
    "😠": "angry",
    "😔": "sad",
    "😍": "loving",
    "😲": "surprised",
    "😱": "shocked",
    "🤔": "thinking",
    "😌": "relaxed",
    "😴": "sleepy",
    "😜": "silly",
    "🙄": "confused",
    "😶": "neutral",
    "🙂": "happy",
    "😆": "laughing",
    "😳": "embarrassed",
    "😉": "winking",
    "😎": "cool",
    "🤤": "delicious",
    "😘": "kissy",
    "😏": "confident",
}
EMOJI_RANGES = [
    (0x1F600, 0x1F64F),
    (0x1F300, 0x1F5FF),
    (0x1F680, 0x1F6FF),
    (0x1F900, 0x1F9FF),
    (0x1FA70, 0x1FAFF),
    (0x2600, 0x26FF),
    (0x2700, 0x27BF),
]




def strip_markdown_for_speech(text):
    """Remove markdown syntax so the synthesiser does not speak it aloud.

    ⚠️ WHY (2026-10-16): Vincent's articulation coach replies are structured
    markdown. The TTS segmenter (core/providers/tts/base.py) only trims
    punctuation from segment ENDS, so the synthesiser received markers
    literally — visible in the server log:

        语音生成成功: **Week 3 Challenge — Impromptu 2-Minute, No Prep**

    Markdown is a DISPLAY format; it exists for the screen, not the ear.

    INVARIANTS (each guarded by a test):
      - never delete words — a cleaner that eats content is worse than one
        that reads a stray asterisk
      - never collapse line structure — the iOS client needs the newlines to
        lay the reply out (that was a separate bug; do not reintroduce it)
      - leave plain prose byte-identical (the common voice case)
      - leave underscores inside words alone (`snake_case_name`)
      - a link speaks its label, never the URL
    """
    if not text:
        return text

    out = text

    # Fenced code blocks: keep the content, drop the fence lines.
    out = re.sub(r"```[^\n]*\n?", "", out)

    # Inline code: drop the backticks, keep the code.
    out = out.replace("`", "")

    # Links: [label](url) -> label. Must run BEFORE emphasis stripping so the
    # label's own markers are handled by the later passes.
    # Linear, not quadratic: `[^\[\]]` cannot backtrack into a nested
    # quantifier, so an unmatched "[" run is single-pass (build-tester measured
    # the previous pattern at 4.1s on 50KB of unmatched brackets).
    out = re.sub(r"\[([^\[\]]*)\]\([^()]*\)", r"\1", out)

    # Images: ![alt](url) -> alt
    out = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", out)

    # Bold / strong: **x** and __x__ (content kept).
    out = re.sub(r"\*\*(.+?)\*\*", r"\1", out, flags=re.DOTALL)
    out = re.sub(r"__(.+?)__", r"\1", out, flags=re.DOTALL)

    # Italic: *x* — but NOT a bullet "* item" at line start.
    # ⚠️ Insert a space when removal would GLUE words together (build-tester:
    # "Try *saying*it slower." became "Try sayingit slower" — a mispronounced
    # word is worse than a stray asterisk). Four cases, because the word
    # character can sit on EITHER side of the marker pair:
    #   *a*b  -> a b   (word after close, none before open at left edge)
    #   a*b*  -> a b
    #   *a*   -> a     (nothing adjacent; no space wanted)
    #   a*b   -> a b
    out = re.sub(r"(?<=\w)\*([^*\n]+?)\*(?=\w)", r" \1 ", out)   # a*b*c
    out = re.sub(r"\*([^*\n]+?)\*(?=\w)", r"\1 ", out)           # *a*b
    out = re.sub(r"(?<=\w)\*([^*\n]+?)\*(?!\*)", r" \1", out)    # a*b*
    out = re.sub(r"(?<!\*)\*(?!\s)([^*\n]+?)\*(?!\*)", r"\1", out)

    # Underscore italic, only at word boundaries so snake_case survives.
    out = re.sub(r"(?<![\w_])_([^_\n]+?)_(?![\w_])", r"\1", out)

    # ATX headings: strip the leading #s and one space.
    out = re.sub(r"^[ \t]*#{1,6}[ \t]*", "", out, flags=re.MULTILINE)

    # Bullet / numbered list markers at line start.
    out = re.sub(r"^[ \t]*[-*+][ \t]+", "", out, flags=re.MULTILINE)
    out = re.sub(r"^[ \t]*\d+[.)][ \t]+", "", out, flags=re.MULTILINE)

    # Blockquote markers.
    out = re.sub(r"^[ \t]*>[ \t]?", "", out, flags=re.MULTILINE)

    # Horizontal rules (--- / *** / ___) on their own line.
    out = re.sub(r"^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$", "", out,
                 flags=re.MULTILINE)

    return out


def get_string_no_punctuation_or_emoji(s):
    """去除字符串首尾的空格、标点符号和表情符号"""
    chars = list(s)
    # 处理开头的字符
    start = 0
    while start < len(chars) and is_punctuation_or_emoji(chars[start]):
        start += 1
    # 处理结尾的字符
    end = len(chars) - 1
    while end >= start and is_punctuation_or_emoji(chars[end]):
        end -= 1
    return "".join(chars[start : end + 1])


def is_punctuation_or_emoji(char):
    """检查字符是否为空格、指定标点或表情符号"""
    # 定义需要去除的中英文标点（包括全角/半角）
    punctuation_set = {
        "，",
        ",",  # 中文逗号 + 英文逗号
        "。",
        ".",  # 中文句号 + 英文句号
        "！",
        "!",  # 中文感叹号 + 英文感叹号
        "“",
        "”",
        '"',  # 中文双引号 + 英文引号
        "：",
        ":",  # 中文冒号 + 英文冒号
        "-",
        "－",  # 英文连字符 + 中文全角横线
        "、",  # 中文顿号
        "[",
        "]",  # 方括号
        "【",
        "】",  # 中文方括号
    }
    if char.isspace() or char in punctuation_set:
        return True
    return is_emoji(char)


async def get_emotion(conn: "ConnectionHandler", text):
    """获取文本内的情绪消息"""
    emoji = "🙂"
    emotion = "happy"
    for char in text:
        if char in EMOJI_MAP:
            emoji = char
            emotion = EMOJI_MAP[char]
            break
    try:
        await conn.websocket.send(
            json.dumps(
                {
                    "type": "llm",
                    "text": emoji,
                    "emotion": emotion,
                    "session_id": conn.session_id,
                }
            )
        )
    except Exception as e:
        conn.logger.bind(tag=TAG).warning(f"发送情绪表情失败，错误:{e}")
    return


def is_emoji(char):
    """检查字符是否为emoji表情"""
    code_point = ord(char)
    return any(start <= code_point <= end for start, end in EMOJI_RANGES)


def check_emoji(text):
    """去除文本中的所有emoji表情（保留换行符，用于屏幕显示时的分段结构）"""
    return "".join(char for char in text if not is_emoji(char))
