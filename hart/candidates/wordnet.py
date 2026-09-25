from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .base import CandidateSelector, CandidateSentence, split_sentences, tokenize_words


class WordNetSelector(CandidateSelector):
    """Select sentences using configurable noun-synset Wu-Palmer similarity."""

    def __init__(
        self,
        threshold: float | None = None,
        wordnet: Any = None,
        synset_strategy: str = "first",
        *,
        candidate_selector_non_re_first_match_only: bool = False,
    ):
        if synset_strategy not in {"first", "max_all", "average", "legacy_asymmetric"}:
            raise ValueError(
                "synset_strategy must be 'first', 'max_all', 'average', or "
                "'legacy_asymmetric'"
            )
        if wordnet is None:
            from nltk.corpus import wordnet as wordnet_corpus
            wordnet = wordnet_corpus
        self.wordnet = wordnet
        self.threshold = threshold
        self.candidate_selector_non_re_first_match_only = candidate_selector_non_re_first_match_only
        self.synset_strategy = synset_strategy

    def _synsets(self, token: str) -> list[Any]:
        return self.wordnet.synsets(token, pos="n")

    def _score(self, sentence: str, query: str) -> float | None:
        sentence_synsets = [self._synsets(token) for token in tokenize_words(sentence.lower())]
        query_synsets = [self._synsets(token) for token in tokenize_words(query.lower())]

        if self.synset_strategy == "first":
            sentence_synsets = [synsets[:1] for synsets in sentence_synsets]
            query_synsets = [synsets[:1] for synsets in query_synsets]

        similarities = [
            score
            for left_synsets in sentence_synsets
            for right_synsets in query_synsets
            for left in left_synsets
            for right in right_synsets
            if (score := left.wup_similarity(right)) is not None
        ]

        if not similarities:
            return None
        if self.synset_strategy == "average":
            return sum(similarities) / len(similarities)
        return max(similarities)

    def _legacy_match(self, sentence: str, query: str) -> float | None:
        """Return the first qualifying historical asymmetric comparison."""
        query_synsets = self._synsets(query.lower())
        if not query_synsets:
            return None

        query_synset = query_synsets[0]
        for token in tokenize_words(sentence.lower()):
            for token_synset in self._synsets(token):
                score = query_synset.wup_similarity(token_synset)
                if score is not None and (
                    self.threshold is None or score >= self.threshold
                ):
                    return float(score)
        return None

    def select(self, text: str, queries: Iterable[str]) -> list[CandidateSentence]:
        queries = list(queries)
        results = []
        for sentence_id, sentence in enumerate(split_sentences(text)):
            if self.synset_strategy == "legacy_asymmetric":
                for query in queries:
                    score = self._legacy_match(sentence, query)
                    if score is not None:
                        results.append(CandidateSentence(sentence_id, sentence, score, query, context_text=sentence))
                        if self.candidate_selector_non_re_first_match_only:
                            return results
                        break
                continue
            matches = [
                (score, query)
                for query in queries
                if (score := self._score(sentence, query)) is not None
            ]
            if not matches:
                continue
            score, query = max(matches)
            if self.threshold is None or score >= self.threshold:
                results.append(CandidateSentence(sentence_id, sentence, score, query, context_text=sentence))
                if self.candidate_selector_non_re_first_match_only:
                    return results
        return results
