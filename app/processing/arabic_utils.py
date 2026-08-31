"""Arabic-aware text normalization.

Most RAG pipelines run a generic Unicode normalizer over every document and
call it done, which for Arabic text tends to silently strip diacritics
(tashkeel) — even though those diacritics can carry meaning (e.g. verb voice,
case ending, or disambiguating two otherwise-identical words). This module
makes that choice explicit and defaults to *preserving* diacritics rather than
stripping them.
"""

import re
import unicodedata

# Arabic diacritics (tashkeel/harakat) live in this Unicode range.
_ARABIC_DIACRITICS = re.compile(
    "["
    "\u0610-\u061a"  # Arabic honorifics / small high marks
    "\u064b-\u065f"  # fatha, damma, kasra, sukun, shadda, etc.
    "\u0670"          # superscript alef
    "\u06d6-\u06dc"
    "\u06df-\u06e8"
    "\u06ea-\u06ed"
    "]"
)

# Tatweel/kashida (ـ) is a justification/decorative elongation character, not
# semantic content — safe to strip unconditionally.
_TATWEEL = "\u0640"

_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_EXTENDED_ARABIC_INDIC_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

_ARABIC_BLOCK = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")


def contains_arabic(text: str) -> bool:
    return bool(_ARABIC_BLOCK.search(text))


def arabic_ratio(text: str) -> float:
    """Fraction of alphabetic characters that fall in the Arabic Unicode block.

    Used by the document analyzer to decide whether Arabic-specific handling
    (diacritic-safe chunk boundaries, RTL-safe splitting) should apply to a
    given block.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    arabic_letters = [ch for ch in letters if _ARABIC_BLOCK.match(ch)]
    return len(arabic_letters) / len(letters)


def strip_diacritics(text: str) -> str:
    """Remove tashkeel. Opt-in — see module docstring for why this isn't default."""
    return _ARABIC_DIACRITICS.sub("", text)


def normalize_arabic(text: str, *, strip_diacritics_: bool = False) -> str:
    """Normalize Arabic text while preserving diacritics unless told otherwise.

    Steps:
    1. Unicode NFC normalization (canonical composition)
    2. Remove tatweel (non-semantic elongation)
    3. Normalize Arabic-Indic digits to ASCII digits (for consistent downstream
       numeric handling, e.g. in tables or benchmark queries)
    4. Optionally strip diacritics (default: no — see module docstring)
    5. Collapse redundant whitespace introduced by the above
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace(_TATWEEL, "")
    text = text.translate(_ARABIC_INDIC_DIGITS).translate(_EXTENDED_ARABIC_INDIC_DIGITS)
    if strip_diacritics_:
        text = strip_diacritics(text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


def safe_split_point(text: str, target_index: int) -> int:
    """Find a chunk boundary near `target_index` that doesn't split a word or,
    critically, separate an Arabic base letter from a following diacritic
    (which changes what the diacritic attaches to when the chunk is re-read
    independently). Searches outward from target_index for whitespace.
    """
    if target_index >= len(text):
        return len(text)

    for offset in range(0, min(80, len(text))):
        for idx in (target_index + offset, target_index - offset):
            if 0 <= idx < len(text) and text[idx].isspace():
                # Don't cut immediately before a diacritic — walk past any
                # diacritics attached to the preceding letter first.
                while idx < len(text) and _ARABIC_DIACRITICS.match(text[idx : idx + 1]):
                    idx += 1
                return idx
    return target_index
