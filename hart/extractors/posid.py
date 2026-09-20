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
    # confirmed that the reconstruction below for female is fine.
    "female": "female",
    "woman": "female",
    "girl": "female",
}
_RACES = ("asian", "black", "white", "hispanic", "latino", "latina", "indian")
_HEIGHT_PATTERNS = (
    r"\b\d+(?:\.\d+)?\s*(?:-?(?:feet?|foot)|ft\.?)\s*(?:\d+\s*(?:inches?|in\.?)?)?\s*(?:tall)?",
    r"\b\d{1,2}\s*['’]\s*\d{1,2}\s*[\"”](?:\s+tall\b)?",
    r"\b(?:short|tall)\b",
)
_BOUNDARIES = {"and", "or", ",", ";", "."}
_IGNORED_DESCRIPTORS = {"a", "an", "the", "wear", "wears", "wearing", "wore"}
_IGNORED_CLOTHING_WORDS = _IGNORED_DESCRIPTORS | {"was", "were", "is", "are", "described", "about"}


class POSIDExtractor(Extractor):
    """Legacy POSID property extraction after modular candidate retrieval."""

    def __init__(
        self,
        candidate_selector: Any,
        queries: Iterable[str] | None = None,
        wordnet: Any = None,
        color_threshold: float = 0.7,
        tagger: Callable[[list[str]], Sequence[tuple[str, str]]] = pos_tag
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
        prior_context: dict[int, list[tuple[str, int, str | None]]] = {}
        recent_shared: tuple[str, ...] = ()
        seen_anchors: list[tuple[str, int, str | None]] = []

        # i.e., if a sentence contains multiple anchors and the word "and", then all anchors are considered shared for that sentence.
        # John and Mike entered the store.
        # recent_shared = ("John", "Mike")
        for sentence_id, sentence in enumerate(sentences):
            anchors = self._person_anchors(sentence)
            prior_context[sentence_id] = list(seen_anchors)

            if len(anchors) > 1 and re.search(r"\band\b", sentence, re.IGNORECASE):
                recent_shared = tuple(anchor[0] for anchor in anchors)
            elif anchors:
                recent_shared = ()

            shared_context[sentence_id] = recent_shared
            seen_anchors.extend(anchors)
        
        seen_candidates: set[tuple[int, str]] = set()
        for candidate in candidates:
            sentence_id, sentence = self._candidate_sentence(candidate, sentences)

            key = (sentence_id, sentence)
            if key in seen_candidates:
                continue
            seen_candidates.add(key)

            ## fixing which query should be used to extract which attribute (fixed vs clothes attributes)
            query = candidate.query.casefold()

            if query == "wearing":
                include_fixed = True
                include_clothing = True

            elif query == "clothes":
                include_fixed = False
                include_clothing = True

            elif query.startswith("regex:"):
                # The fixed REGEX query
                include_fixed = True
                include_clothing = False

            else:
                raise ValueError(f"Unknown candidate query role: {candidate.query!r}")

            mentions.extend(
                self._extract_sentence(
                    sentence,
                    sentence_id,
                    include_fixed=include_fixed,
                    include_clothing=include_clothing,
                    shared_anchors=shared_context.get(sentence_id, ()),
                    prior_anchors=prior_context.get(sentence_id, []),
                )
            )
        return resolve_mentions(document_id, mentions)

    @staticmethod
    def _clothing_portion(sentence: str) -> tuple[str, int] | None:
        match = re.search(
            r"\b(?:wear|wears|wearing|wore)\b[^.]*",
            sentence,
            re.IGNORECASE,
        )
        if match is None:
            return None

        return match.group(0), match.start()

    @staticmethod
    def _candidate_sentence(
        candidate: CandidateSentence,
        sentences: list[str],
    ) -> tuple[int, str]:
        sentence = candidate.context_text or candidate.text
        if (
            candidate.sentence_id > 0
            and re.match(r"\s*(?:wear|wears|wearing|wore)\b", candidate.text, re.IGNORECASE)
            and candidate.sentence_id < len(sentences)
            and sentences[candidate.sentence_id - 1] not in sentence
            and any(
                anchor_role is not None
                for _, _, anchor_role in find_person_anchors(
                    sentences[candidate.sentence_id - 1]
                )
            )
        ):
            sentence = f"{sentences[candidate.sentence_id - 1]} {sentence}"
        return candidate.sentence_id, sentence
    
    def _extract_sentence(
        self,
        sentence: str,
        sentence_id: int,
        *,
        include_fixed: bool,
        include_clothing: bool,
        shared_anchors: tuple[str, ...] = (),
        prior_anchors: list[tuple[str, int, str | None]] | None = None,
    ) -> list[AttributeMention]:
        mentions: list[AttributeMention] = []
        anchors = self._person_anchors(sentence)
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
            clothing_start = self._clothing_portion(sentence)
            for race in _RACES:
                for match in re.finditer(rf"\b{race}\b", sentence, re.IGNORECASE):
                    if clothing_start is not None and match.start() >= clothing_start[1]:
                        continue
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
                    value = match.group(0).strip()
                    if pattern_index == 1:
                        value = re.sub(r"\s+tall$", "", value, flags=re.IGNORECASE)
                    mentions.append(
                        self._anchored_mention(
                            "height",
                            value,
                            sentence_id,
                            match.start(),
                            anchors,
                        )
                    )
                    height_spans.append(match.span())

        if include_clothing:
            portion = self._clothing_portion(sentence)
            if portion is None:
                fixed_spans = self._fixed_property_spans(sentence)
                clothing_text = self._without_spans(sentence, fixed_spans)
                clothing_offset = 0
            else:
                clothing_text, clothing_offset = portion

            tokens = tokenize_words(clothing_text)
            tagged = list(self.tagger(tokens))
            token_positions = [
                clothing_offset + position
                for position in self._token_positions(clothing_text, tokens)
            ]
            excluded_words = {
                anchor[0].casefold()
                for anchor in anchors
            }
            for start, _, item, descriptors in self._clothing_spans(
                tokens,
                tagged,
                excluded_words=excluded_words,
            ):
                position = token_positions[start]
                pronoun_match = re.search(
                    r"\b(?:he|she|they)\b",
                    sentence[:position],
                    re.IGNORECASE,
                )
                if pronoun_match and not anchors and prior_anchors:
                    mentions.extend(
                        AttributeMention(
                            "__anchor__", None, sentence_id, anchor_position, anchor, role
                        )
                        for anchor, anchor_position, role in prior_anchors
                    )
                mention_anchors = anchors
                if pronoun_match and not mention_anchors and prior_anchors:
                    mention_anchors = prior_anchors

                # adding this for the one special case: jacket/hoodie, where the annotator labeled both items as a single item. 
                # This is a hacky solution, but it works for this case.
                # So posid extraction was correct, but annotator issue.
                # Would be insteresting to see if LLMs can recognize them as a single item.
                item_names = [
                    name.strip()
                    for name in item.split("/")
                    if name.strip()
                ]

                for item_name in item_names:

                    # the following part was already part of code without the jacket/ hoodie addition
                    mentions.append(
                        self._anchored_mention(
                            "wearing",
                            {
                                "item": item_name,
                                "descriptors": list(descriptors),
                            },
                            sentence_id,
                            position,
                            mention_anchors,
                            pronoun=(
                                pronoun_match.group(0).lower()
                                if pronoun_match
                                else None
                            ),
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

    def _clothing_spans(
        self,
        tokens: list[str],
        tagged: list[tuple[str, str]],
        *,
        excluded_words: set[str] | None = None,
    ) -> list[tuple[int, int, str, list[str]]]:
        """Build open-vocabulary property names from noun runs."""
        excluded_words = excluded_words or set()
        spans: list[tuple[int, int, str, list[str]]] = []
        descriptors: list[str] = []
        noun_start: int | None = None
        noun_tokens: list[str] = []

        def flush_nouns() -> None:
            nonlocal noun_start, noun_tokens, descriptors
            if noun_start is not None and noun_tokens:
                spans.append(
                    (
                        noun_start,
                        noun_start + len(noun_tokens),
                        " ".join(noun_tokens),
                        list(descriptors),
                    )
                )
            noun_start = None
            noun_tokens = []
            descriptors = []

        for index, (token, tag) in enumerate(tagged):
            lowered = token.casefold()
            is_noun = tag.startswith("NN")
            is_adjective = tag.startswith("JJ")
            if lowered in _BOUNDARIES or lowered in {"both", "but"}:
                if noun_start is not None:
                    flush_nouns()
                continue
            if is_adjective:
                descriptors.append(lowered)
                continue
            if is_noun:
                if lowered in excluded_words or lowered in _IGNORED_CLOTHING_WORDS:
                    flush_nouns()
                    continue
                if self._is_color(lowered):
                    flush_nouns()
                    descriptors.append(lowered)
                    continue
                if noun_start is None:
                    noun_start = index
                noun_tokens.append(lowered)
                continue
            if tag in {"CC", "DT", "IN", ":", "-", "(`", "`)"}:
                if noun_start is not None:
                    flush_nouns()
                continue
            flush_nouns()

        flush_nouns()
        return spans

    def _legacy_clothing_spans(
        self,
        tokens: list[str],
        tagged: list[tuple[str, str]],
        *,
        excluded_words: set[str] | None = None,
    ) -> list[tuple[int, int, str, list[str]]]:
        """
        Reproduce the conservative 2021 POSID clothing parser.

        The historical parser stopped after unexpected POS patterns instead
        of continuing through the complete candidate sentence. This reduces
        unrelated nouns being interpreted as clothing.
        """
        excluded_words = excluded_words or set()

        spans: list[tuple[int, int, str, list[str]]] = []
        adjectives: list[str] = []
        append_indices: list[int] = []

        for index, (token, tag) in enumerate(tagged):
            lowered = token.casefold()

            # The 2021 notebook always ignored the first candidate token.
            # For Regex candidates this was normally "wearing".
            if index == 0:
                continue

            # Historical handling of:
            #     "She had ..."
            #     "He was wearing ..."
            if tag.startswith("VBD") and index == 1:
                continue

            if (
                tag.startswith("VBG")
                and index == 2
                and tagged[index - 1][1].startswith("VBD")
            ):
                continue

            is_connector = (
                tag.startswith("DT")
                or tag.startswith("CC")
                or tag.startswith("IN")
                or tag.startswith("VBD")
                or tag.startswith("RP")
                or tag.startswith("RB")
                or len(tag) == 1
            )

            if is_connector:
                previous_tag = tagged[index - 1][1]

                # Reproduce the old early-stop condition.
                if (
                    tag.startswith(("VBD", "RP", "RB"))
                    and (
                        len(previous_tag) == 1
                        or previous_tag.startswith("PRP")
                    )
                ):
                    break

                append_indices.append(index)
                continue

            if tag.startswith("JJ"):
                append_indices = []
                adjectives.append(lowered)
                continue

            if tag.startswith("NN"):
                if (
                    lowered in excluded_words
                    or lowered in _IGNORED_CLOTHING_WORDS
                ):
                    append_indices = []
                    adjectives = []
                    continue

                # The historical code treated color-like nouns as descriptors.
                if self._is_color(lowered):
                    append_indices = []
                    adjectives.append(lowered)
                    continue

                item = lowered
                item_start = index

                # Preserve historical items such as "buttoned up shirt".
                for append_index in reversed(append_indices):
                    append_token, append_tag = tagged[append_index]

                    if append_tag.startswith(("VBD", "RP", "RB")):
                        item = f"{append_token.casefold()} {item}"
                        item_start = append_index
                    else:
                        break

                # Combine consecutive nouns into one item, such as:
                #     "tank top"
                #     "dress pants"
                if (
                    index > 0
                    and tagged[index - 1][1].startswith("NN")
                    and spans
                    and spans[-1][2].endswith(
                        tagged[index - 1][0].casefold()
                    )
                ):
                    previous_start, _, previous_item, previous_descriptors = spans[-1]

                    spans[-1] = (
                        previous_start,
                        index + 1,
                        f"{previous_item} {lowered}",
                        previous_descriptors,
                    )
                else:
                    spans.append(
                        (
                            item_start,
                            index + 1,
                            item,
                            list(adjectives),
                        )
                    )

                adjectives = []
                append_indices = []
                continue

            # The key historical behavior: stop instead of continuing to
            # search for nouns elsewhere in the sentence.
            break

        return spans

    @staticmethod
    def _person_anchors(sentence: str) -> list[tuple[str, int, str | None]]:
        return [
            anchor
            for anchor in find_person_anchors(sentence)
            if anchor[0].casefold() not in _RACES
        ]

    @staticmethod
    def _fixed_property_spans(sentence: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        for word in _GENDERS:
            spans.extend(
                match.span()
                for match in re.finditer(rf"\b{word}\b", sentence, re.IGNORECASE)
            )
        clothing_start = POSIDExtractor._clothing_portion(sentence)
        for race in _RACES:
            spans.extend(
                match.span()
                for match in re.finditer(rf"\b{race}\b", sentence, re.IGNORECASE)
                if clothing_start is None or match.start() < clothing_start[1]
            )
        height_spans: list[tuple[int, int]] = []
        for pattern_index, pattern in enumerate(_HEIGHT_PATTERNS):
            for match in re.finditer(pattern, sentence, re.IGNORECASE):
                if pattern_index > 0 and any(
                    match.start() >= start and match.end() <= end
                    for start, end in height_spans
                ):
                    continue
                spans.append(match.span())
                height_spans.append(match.span())
        return spans

    @staticmethod
    def _without_spans(sentence: str, spans: list[tuple[int, int]]) -> str:
        characters = list(sentence)
        for start, end in spans:
            characters[start:end] = [" "] * (end - start)
        return "".join(characters)

    def _is_color(self, token: str) -> bool:
        color_synsets = self.wordnet.synsets("color", pos="n")
        if not color_synsets:
            return False
        for token_synset in self.wordnet.synsets(token, pos="n"):
            score = color_synsets[0].wup_similarity(token_synset)
            if score is not None and score > self.color_threshold:
                return True
        return False

    
