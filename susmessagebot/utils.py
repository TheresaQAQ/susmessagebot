"""Text helpers shared by classification and retrieval."""

from __future__ import annotations

import re
import unicodedata

_DIGIT_WORDS = {
    "ZERO": "0",
    "ONE": "1",
    "TWO": "2",
    "THREE": "3",
    "FOUR": "4",
    "FIVE": "5",
    "SIX": "6",
    "SEVEN": "7",
    "EIGHT": "8",
    "NINE": "9",
}

_ZERO_WIDTH = dict.fromkeys(
    map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"),
    None,
)


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x3040 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7AF
        or 0xF900 <= code <= 0xFAFF
    )


def _fold_char(ch: str) -> str:
    if ch.isascii() or _is_cjk(ch):
        return ch
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return ch
    if name.startswith("MATHEMATICAL "):
        token = name.split()[-1]
        if token in _DIGIT_WORDS:
            return _DIGIT_WORDS[token]
        if len(token) == 1 and token.isalpha():
            return token.lower() if "SMALL" in name else token
    decomp = unicodedata.normalize("NFKD", ch)
    base = "".join(c for c in decomp if not unicodedata.combining(c))
    return base or ch


def normalize_text(text: str) -> str:
    """
    Fold evasion-oriented Unicode while preserving CJK.

    - NFKC: fullwidth Latin/digits → ASCII
    - Drop zero-width / soft-hyphen characters
    - Fold mathematical alphanumeric symbols to plain Latin/digits
    - Strip combining marks on non-CJK characters
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    folded = "".join(_fold_char(ch) for ch in text)
    return re.sub(r"\s+", " ", folded).strip()
