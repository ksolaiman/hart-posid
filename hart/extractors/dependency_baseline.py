from __future__ import annotations

import re
from typing import Any

from .base import Extractor, PredictionIndex
from .person_resolution import (
    AttributeMention,
    find_person_anchors,
    nearest_preceding_anchor,
    resolve_mentions,
)
from .regex_baseline import CLOTHING_ITEMS, GENDER_WORDS


_WEAR_LEMMAS = {"wear", "wears", "wearing", "wore"}
_COLOR_WORDS = {
    "black", "blue", "brown", "gray", "grey", "green", "orange", "pink",
    "purple", "red", "white", "yellow",
}


class DependencyExtractor(Extractor):
    """Small spaCy dependency-rule baseline with an injected pipeline."""

    def __init__(self, nlp: Any):
        self.nlp = nlp

    def extract(self, text: str, document_id: str = "document") -> PredictionIndex:
        doc = self.nlp(text)
        mentions: list[AttributeMention] = []
        recent_shared: tuple[str, ...] = ()
        for sentence_id, sentence in enumerate(doc.sents):
            sentence_text = sentence.text
            anchors = find_person_anchors(sentence_text)
            if len(anchors) > 1 and re.search(r"\band\b", sentence_text, re.IGNORECASE):
                recent_shared = tuple(anchor[0] for anchor in anchors)
            for anchor, position, role in anchors:
                mentions.append(
                    AttributeMention("__anchor__", None, sentence_id, position, anchor, role)
                )
            tokens = list(sentence)
            sentence_start = int(getattr(sentence, "start_char", 0))
            for token in tokens:
                token_text = str(getattr(token, "text", ""))
                lowered = token_text.lower()
                position = int(getattr(token, "idx", 0)) - sentence_start
                local_anchor = nearest_preceding_anchor(anchors, position)
                anchor = local_anchor[0] if local_anchor else None
                role = local_anchor[2] if local_anchor else None
                if lowered in GENDER_WORDS:
                    mentions.append(
                        AttributeMention(
                            "gender",
                            GENDER_WORDS[lowered],
                            sentence_id,
                            position,
                            anchor,
                            role,
                        )
                    )
                if lowered in CLOTHING_ITEMS and self._in_wearing_relation(token, tokens):
                    descriptors = []
                    for child in getattr(token, "children", ()):
                        child_text = str(getattr(child, "text", "")).lower()
                        if (
                            getattr(child, "dep_", "") == "amod"
                            or getattr(child, "pos_", "") == "ADJ"
                            or child_text in _COLOR_WORDS
                        ):
                            descriptors.append(child_text)
                    item = lowered
                    if item.endswith("s") and item not in {"pants", "jeans", "shoes", "shorts"}:
                        item = item[:-1]
                    pronoun = self._preceding_pronoun(tokens, token)
                    shared = (
                        recent_shared
                        if re.search(r"\bboth\b", sentence_text, re.IGNORECASE)
                        else ()
                    )
                    mentions.append(
                        AttributeMention(
                            "wearing",
                            {"item": item, "descriptors": descriptors},
                            sentence_id,
                            position,
                            anchor,
                            role,
                            pronoun,
                            shared,
                        )
                    )
                if getattr(token, "like_num", False):
                    following = " ".join(
                        str(getattr(item, "text", ""))
                        for item in tokens
                        if getattr(item, "idx", 0) >= getattr(token, "idx", 0)
                    )
                    height_match = re.match(
                        r"\d+(?:\.\d+)?\s*(?:-?(?:feet?|ft))"
                        r"(?:\s+\d+(?:\.\d+)?\s*(?:inches?|in))?"
                        r"(?:\s+tall)?",
                        following,
                        re.IGNORECASE,
                    )
                    if height_match and re.search(r"(?:feet?|ft|['’]|inches?|in|tall)", height_match.group(0), re.IGNORECASE):
                        mentions.append(
                            AttributeMention(
                                "height",
                                height_match.group(0),
                                sentence_id,
                                position,
                                anchor,
                                role,
                            )
                        )
        return resolve_mentions(document_id, mentions)

    @staticmethod
    def _in_wearing_relation(token: Any, tokens: list[Any]) -> bool:
        related = [getattr(token, "head", None), *getattr(token, "ancestors", ())]
        if any(
            str(getattr(item, "lemma_", getattr(item, "text", ""))).lower()
            in _WEAR_LEMMAS
            for item in related
            if item is not None
        ):
            return True
        # Deterministic test Docs may omit parser-only head/ancestor properties.
        return any(
            str(getattr(item, "lemma_", getattr(item, "text", ""))).lower()
            in _WEAR_LEMMAS
            for item in tokens
        )

    @staticmethod
    def _preceding_pronoun(tokens: list[Any], clothing_token: Any) -> str | None:
        clothing_position = int(getattr(clothing_token, "idx", 0))
        pronouns = [
            str(getattr(token, "text", "")).lower()
            for token in tokens
            if int(getattr(token, "idx", 0)) < clothing_position
            and str(getattr(token, "text", "")).lower() in {"he", "she", "they"}
        ]
        return pronouns[-1] if pronouns else None
