from __future__ import annotations

import re
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from nltk.corpus import wordnet


# Plural-only garment names retain their lexical form, even when WordNet also
# offers a singular with a different sense (e.g. jeans -> jean, a fabric).
_PLURAL_CLOTHING = frozenset({
    "pants", "jeans", "shorts", "trousers", "tights", "leggings",
    "overalls", "dungarees", "briefs", "underpants", "clothes", "glasses",
})


@lru_cache(maxsize=4096)
def canonicalize_clothing_item(value: str | None) -> str | None:
    """Normalize item heads symmetrically; descriptors and semantic aliases stay out.

    Fold case/whitespace, strip boundary punctuation, and treat hyphens as
    spaces. Prefer WordNet noun morphology for the final word. Only when
    WordNet has no lemma, remove a regular final s, excluding common singular
    endings ss/us/is. Plural-only garments are explicitly protected.
    """
    if value is None:
        return None
    text = " ".join(str(value).strip().casefold().split())
    if not text:
        return text
    text = text.strip(" .,:;!?\"'()[]{}")
    words = re.sub(r"[-‐‑‒–—]+", " ", text).split()
    if not words:
        return ""
    head = words[-1]
    if head not in _PLURAL_CLOTHING:
        lemma = wordnet.morphy(head, wordnet.NOUN)
        if lemma is not None:
            head = lemma
        elif len(head) > 3 and head.endswith("s") and not head.endswith(("ss", "us", "is")):
            head = head[:-1]
    words[-1] = head
    return " ".join(words)


_HEIGHT_QUOTE_TRANSLATION = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
})


def normalize_height_surface(value: str) -> str:
    """Canonicalize only equivalent punctuation/spacing in height surfaces."""
    normalized = " ".join(value.translate(_HEIGHT_QUOTE_TRANSLATION).strip().split())
    if re.search(r"\d", normalized):
        normalized = re.sub(r"\s+tall$", "", normalized, flags=re.IGNORECASE)
    return normalized


def normalize_legacy_value(value: Any) -> Any:
    """Conservatively replace legacy string nulls throughout a value."""
    if value == "null":
        return None
    if isinstance(value, Mapping):
        return {key: normalize_legacy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_legacy_value(item) for item in value]
    return value


def normalize_legacy_wearing(value: Any) -> list[dict[str, Any]]:
    """Convert arbitrary legacy clothing keys into stable item records."""
    normalized = normalize_legacy_value(value)
    if not isinstance(normalized, Mapping):
        raise ValueError("Legacy wearing annotations must be objects")
    return [
        {"item": item, "descriptors": descriptors}
        for item, descriptors in normalized.items()
    ]
