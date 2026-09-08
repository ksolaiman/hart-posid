from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from nltk import pos_tag

from hart.candidates import CandidateSentence, split_sentences, tokenize_words

from .base import Extractor, PredictionIndex
from .person_resolution import (
    AttributeMention,
    find_person_anchors,
    nearest_preceding_anchor,
    resolve_mentions,
)


_GENDERS = {
    "male": "male",
    "man": "male",
    "boy": "male",
    # These symmetric forms are conservative reconstructions; the surviving
    # historical notebook only demonstrates the male-specific rules.
    # confirmed that the reconstruction below for female if fine. 
    "female": "female",
    "woman": "female",
    "girl": "female",
}
_RACES = ("asian", "black", "white", "hispanic", "latino", "latina")
_HEIGHT_PATTERNS = (
    r"\b\d+(?:\.\d+)?\s*(?:-?(?:feet?|foot)|ft\.?)\s*(?:\d+\s*(?:inches?|in\.?)?)?\s*(?:tall)?",
    r"\b\d{1,2}\s*['’]\s*\d{1,2}\s*[\"”]",
    r"\b(?:short|tall)\b",
)
_CLOTHING_NAMES = (
    ("baseball", "cap"),
    ("tee", "shirt"),
    ("t", "shirt"),
    ("jacket",),
    ("coat",),
    ("shirt",),
    ("pants",),
    ("jeans",),
    ("shoes",),
    ("dress",),
    ("hat",),
    ("cap",),
    ("sweater",),
    ("shorts",),
    ("skirt",),
)
_BOUNDARIES = {"and", "or", ",", ";", "."}
_IGNORED_DESCRIPTORS = {"a", "an", "the", "wear", "wears", "wearing", "wore"}


class POSIDExtractor(Extractor):
    """Legacy POSID property extraction after modular candidate retrieval."""

    def __init__(
        self,
        candidate_selector: Any,
        queries: Iterable[str] | None = None,
        wordnet: Any = None,
        color_threshold: float = 0.7,
        tagger: Callable[[list[str]], Sequence[tuple[str, str]]] = pos_tag,
    ):
        if wordnet is None:
            from nltk.corpus import wordnet as wordnet_corpus

            wordnet = wordnet_corpus
        self.candidate_selector = candidate_selector
        self.queries = list(queries) if queries is not None else []
        self.wordnet = wordnet
        self.color_threshold = color_threshold
        self.tagger = tagger

    def extract(self, text: str, document_id: str = "document") -> PredictionIndex:
        candidates = self.candidate_selector.select(text, self.queries)
        if not candidates:
            return {document_id: []}

        sentences = split_sentences(text)
        mentions: list[AttributeMention] = []
        shared_context: dict[int, tuple[str, ...]] = {}
        recent_shared: tuple[str, ...] = ()
        # Fixed REGEXPROPVAL properties remain document-level once retrieval
        # succeeds; variable clothing extraction is candidate-scoped below.
        for sentence_id, sentence in enumerate(sentences):
            sentence_anchors = find_person_anchors(sentence)
            if len(sentence_anchors) > 1 and re.search(
                r"\band\b", sentence, re.IGNORECASE
            ):
                recent_shared = tuple(anchor[0] for anchor in sentence_anchors)
            shared_context[sentence_id] = recent_shared
            mentions.extend(
                self._extract_sentence(
                    sentence,
                    sentence_id,
                    include_fixed=True,
                    include_clothing=False,
                )
            )

        seen_sentence_ids: set[int] = set()
        for candidate in candidates:
            sentence_id, sentence = self._candidate_sentence(candidate, sentences)
            if sentence_id in seen_sentence_ids:
                continue
            seen_sentence_ids.add(sentence_id)
            mentions.extend(
                self._extract_sentence(
                    sentence,
                    sentence_id,
                    include_fixed=False,
                    include_clothing=True,
                    shared_anchors=shared_context.get(sentence_id, ()),
                )
            )
        return resolve_mentions(document_id, mentions)

    @staticmethod
    def _candidate_sentence(
        candidate: CandidateSentence,
        sentences: list[str],
    ) -> tuple[int, str]:
        if 0 <= candidate.sentence_id < len(sentences):
            return candidate.sentence_id, sentences[candidate.sentence_id]
        return candidate.sentence_id, candidate.text

    def _extract_sentence(
        self,
        sentence: str,
        sentence_id: int,
        *,
        include_fixed: bool,
        include_clothing: bool,
        shared_anchors: tuple[str, ...] = (),
    ) -> list[AttributeMention]:
        mentions: list[AttributeMention] = []
        anchors = find_person_anchors(sentence)
        if include_fixed:
            for anchor, position, role in anchors:
                mentions.append(
                    AttributeMention("__anchor__", None, sentence_id, position, anchor, role)
                )

            for word, value in _GENDERS.items():
                for match in re.finditer(rf"\b{word}\b", sentence, re.IGNORECASE):
                    mentions.append(
                        self._anchored_mention(
                            "gender", value, sentence_id, match.start(), anchors
                        )
                    )

            # This small symmetric race lexicon is reconstructed from task labels;
            # it extracts explicit text only and never infers race from a name.
            for race in _RACES:
                for match in re.finditer(rf"\b{race}\b", sentence, re.IGNORECASE):
                    mentions.append(
                        self._anchored_mention(
                            "race", race, sentence_id, match.start(), anchors
                        )
                    )

            height_spans: list[tuple[int, int]] = []
            for pattern_index, pattern in enumerate(_HEIGHT_PATTERNS):
                for match in re.finditer(pattern, sentence, re.IGNORECASE):
                    if pattern_index > 0 and any(
                        match.start() >= start and match.end() <= end
                        for start, end in height_spans
                    ):
                        continue
                    mentions.append(
                        self._anchored_mention(
                            "height",
                            match.group(0).strip(),
                            sentence_id,
                            match.start(),
                            anchors,
                        )
                    )
                    height_spans.append(match.span())

        if include_clothing:
            tokens = tokenize_words(sentence)
            tagged = list(self.tagger(tokens))
            token_positions = self._token_positions(sentence, tokens)
            for start, _, item in self._clothing_spans(tokens):
                position = token_positions[start]
                pronoun_match = re.search(
                    r"\b(?:he|she|they)\b",
                    sentence[:position],
                    re.IGNORECASE,
                )
                mentions.append(
                    self._anchored_mention(
                        "wearing",
                        {"item": item, "descriptors": self._descriptors(tagged, start)},
                        sentence_id,
                        position,
                        anchors,
                        pronoun=pronoun_match.group(0).lower() if pronoun_match else None,
                        shared_anchors=(
                            shared_anchors
                            if re.search(r"\bboth\b", sentence, re.IGNORECASE)
                            else ()
                        ),
                    )
                )
        return mentions

    @staticmethod
    def _anchored_mention(
        attribute: str,
        value: Any,
        sentence_id: int,
        position: int,
        anchors: list[tuple[str, int, str | None]],
        pronoun: str | None = None,
        shared_anchors: tuple[str, ...] = (),
    ) -> AttributeMention:
        local = nearest_preceding_anchor(anchors, position)
        return AttributeMention(
            attribute,
            value,
            sentence_id,
            position,
            local[0] if local else None,
            local[2] if local else None,
            pronoun,
            shared_anchors,
        )

    @staticmethod
    def _token_positions(sentence: str, tokens: list[str]) -> list[int]:
        positions: list[int] = []
        cursor = 0
        lowered = sentence.lower()
        for token in tokens:
            position = lowered.find(token.lower(), cursor)
            if position < 0:
                position = cursor
            positions.append(position)
            cursor = position + len(token)
        return positions

    @staticmethod
    def _clothing_spans(tokens: list[str]) -> list[tuple[int, int, str]]:
        lowered = [token.lower() for token in tokens]
        spans: list[tuple[int, int, str]] = []
        index = 0
        while index < len(tokens):
            matched = next(
                (
                    name
                    for name in _CLOTHING_NAMES
                    if tuple(lowered[index : index + len(name)]) == name
                ),
                None,
            )
            if matched is None:
                index += 1
                continue
            spans.append((index, index + len(matched), " ".join(matched)))
            index += len(matched)
        return spans

    def _descriptors(
        self,
        tagged: list[tuple[str, str]],
        item_start: int,
    ) -> list[str]:
        descriptors: list[str] = []
        index = item_start - 1
        while index >= 0:
            token, tag = tagged[index]
            lowered = token.lower()
            if lowered in _BOUNDARIES:
                break
            if lowered not in _IGNORED_DESCRIPTORS and (
                tag.startswith("JJ") or tag.startswith("NN")
            ):
                descriptors.append(lowered)
            index -= 1
        descriptors.reverse()
        return [token for token in descriptors if self._is_color(token)]

    def _is_color(self, token: str) -> bool:
        color_synsets = self.wordnet.synsets("color", pos="n")
        if not color_synsets:
            return False
        for token_synset in self.wordnet.synsets(token, pos="n"):
            score = color_synsets[0].wup_similarity(token_synset)
            if score is not None and score > self.color_threshold:
                return True
        return False
