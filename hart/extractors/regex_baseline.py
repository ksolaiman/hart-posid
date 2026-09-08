from __future__ import annotations

import re

from nltk.tokenize import sent_tokenize

from .base import Extractor, PredictionIndex
from .person_resolution import (
    AttributeMention,
    find_person_anchors,
    nearest_preceding_anchor,
    resolve_mentions,
)

GENDER_WORDS = {
    "male": "male",
    "man": "male",
    "boy": "male",
    "female": "female",
    "woman": "female",
    "girl": "female",
}
CLOTHING_ITEMS = (
    "jacket", "jackets", "coat", "coats", "shirt", "shirts", "pants",
    "jeans", "shoes", "dress", "dresses", "hat", "hats", "sweater",
    "sweaters", "shorts", "skirt", "skirts",
)
COLORS = (
    "black", "blue", "brown", "gray", "grey", "green", "orange", "pink",
    "purple", "red", "white", "yellow",
)


def surface_mentions(text: str) -> list[AttributeMention]:
    mentions: list[AttributeMention] = []
    recent_shared: tuple[str, ...] = ()
    for sentence_id, sentence in enumerate(sent_tokenize(text)):
        anchors = find_person_anchors(sentence)
        if len(anchors) > 1 and re.search(r"\band\b", sentence, re.IGNORECASE):
            recent_shared = tuple(anchor[0] for anchor in anchors)
        for anchor, position, role in anchors:
            mentions.append(AttributeMention("__anchor__", None, sentence_id, position, anchor, role))

        for word, value in GENDER_WORDS.items():
            for match in re.finditer(rf"\b{word}\b", sentence, re.IGNORECASE):
                local_anchor = nearest_preceding_anchor(anchors, match.start())
                anchor = local_anchor[0] if local_anchor else None
                role = local_anchor[2] if local_anchor else None
                mentions.append(AttributeMention("gender", value, sentence_id, match.start(), anchor, role))

        height_patterns = (
            r"\b\d+(?:\.\d+)?\s*(?:-?(?:feet?|foot)|ft\.?)\s*(?:\d+\s*(?:inches?|in\.?)?)?\s*(?:tall)?",
            r"\b\d{1,2}\s*['’]\s*\d{1,2}\s*[\"”]",
            r"\b(?:short|tall)\b",
        )
        for pattern in height_patterns:
            for match in re.finditer(pattern, sentence, re.IGNORECASE):
                local_anchor = nearest_preceding_anchor(anchors, match.start())
                mentions.append(AttributeMention("height", match.group(0).strip(), sentence_id, match.start(), local_anchor[0] if local_anchor else None, local_anchor[2] if local_anchor else None))

        clothing_match = re.search(r"\b(?:wear|wears|wearing|wore)\b(?P<clothing>[^.]+)", sentence, re.IGNORECASE)
        if clothing_match:
            clothing = clothing_match.group("clothing")
            pronoun_match = re.search(r"\b(?:he|she|they)\b", sentence[: clothing_match.start()], re.IGNORECASE)
            shared_anchors = recent_shared if re.search(r"\bboth\b", sentence, re.IGNORECASE) else ()
            for item_match in re.finditer(
                rf"(?P<descriptor>[A-Za-z -]*?)\b(?P<item>{'|'.join(CLOTHING_ITEMS)})\b",
                clothing,
                re.IGNORECASE,
            ):
                descriptor_text = item_match.group("descriptor").strip()
                descriptors = [
                    color for color in COLORS if re.search(rf"\b{color}\b", descriptor_text, re.IGNORECASE)
                ]
                item = item_match.group("item").lower()
                if item.endswith("s") and item not in {"pants", "jeans", "shoes", "shorts"}:
                    item = item[:-1]
                item_position = clothing_match.start() + item_match.start("item")
                local_anchor = nearest_preceding_anchor(anchors, item_position)
                mentions.append(
                    AttributeMention(
                        "wearing",
                        {"item": item, "descriptors": descriptors},
                        sentence_id,
                        item_position,
                        local_anchor[0] if local_anchor else None,
                        local_anchor[2] if local_anchor else None,
                        pronoun_match.group(0).lower() if pronoun_match else None,
                        shared_anchors,
                        {"resolution": "pronoun"} if pronoun_match else None,
                    )
                )
    return mentions


class RegexBaseline(Extractor):
    """Direct 2026 surface baseline over raw document text."""

    def extract(self, text: str, document_id: str = "document") -> PredictionIndex:
        return resolve_mentions(document_id, surface_mentions(text))
