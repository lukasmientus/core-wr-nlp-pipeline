"""Lightweight preprocessing utilities (language-agnostic by design)."""

from __future__ import annotations
import re
from typing import List


_SENT_SPLIT_RE = re.compile(
    r"""            # very lightweight sentence segmentation
    (?<=[.!?])         # end punctuation
    \s+               # whitespace
    (?=(?:\"|\'|\(|\[)?[A-ZÄÖÜ0-9])  # next sentence likely starts with cap/number
    """,
    re.VERBOSE,
)


def split_reflection_into_sentences(text: str) -> List[str]:
    """Split a reflection text into sentences using punctuation-based heuristics.

    This intentionally avoids heavy NLP preprocessing to keep the pipeline transferable.

    Parameters
    ----------
    text:
        Raw reflection text.

    Returns
    -------
    List[str]
        Non-empty sentence strings.
    """
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return []

    # Split and keep non-empty parts
    parts = re.split(_SENT_SPLIT_RE, cleaned)
    sentences = [p.strip() for p in parts if p and p.strip()]
    return sentences
