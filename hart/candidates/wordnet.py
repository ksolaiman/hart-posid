from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .base import CandidateSelector, CandidateSentence, split_sentences, tokenize_words


class WordNetSelector(CandidateSelector):
    """Select sentences using the highest WordNet path similarity."""

    def __init__(self, threshold: float | None = None, wordnet: Any = None, pos: str | None = None):
        if wordnet is None:
            from nltk.corpus import wordnet as wordnet_corpus
            wordnet = wordnet_corpus
        self.wordnet = wordnet
        self.threshold = threshold
        self.pos = pos

    def _synsets(self, token: str) -> list[Any]:
        return self.wordnet.synsets(token, pos=self.pos) if self.pos else self.wordnet.synsets(token)

    def _score(self, sentence: str, query: str) -> float:
        sentence_synsets = [synset for token in tokenize_words(sentence.lower()) for synset in self._synsets(token)]
        query_synsets = [synset for token in tokenize_words(query.lower()) for synset in self._synsets(token)]
        similarities = [left.path_similarity(right) or 0.0 for left in sentence_synsets for right in query_synsets]
        return max(similarities, default=0.0)

    def select(self, text: str, queries: Iterable[str]) -> list[CandidateSentence]:
        queries = list(queries)
        results = []
        for sentence_id, sentence in enumerate(split_sentences(text)):
            matches = [(self._score(sentence, query), query) for query in queries]
            if not matches:
                continue
            score, query = max(matches)
            if self.threshold is None or score >= self.threshold:
                results.append(CandidateSentence(sentence_id, sentence, score, query))
        return results