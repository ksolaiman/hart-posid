from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any

from .base import CandidateSelector, CandidateSentence, split_sentences, tokenize_words


class Word2VecSelector(CandidateSelector):
    """Select sentences whose mean word vector is close to a query vector."""

    def __init__(self, model: Any, threshold: float | None = None):
        self.model = model.wv if hasattr(model, "wv") else model
        self.threshold = threshold

    def _vector(self, tokens: list[str]) -> list[float] | None:
        vectors = []
        for token in tokens:
            try:
                vectors.append(self.model[token])
            except (KeyError, TypeError):
                continue
        if not vectors:
            return None
        return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(len(vectors[0]))]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(sum(value * value for value in right))
        return sum(left[index] * right[index] for index in range(len(left))) / denominator if denominator else 0.0

    def select(self, text: str, queries: Iterable[str]) -> list[CandidateSentence]:
        query_vectors = [(query, self._vector([token.lower() for token in tokenize_words(query)])) for query in queries]
        query_vectors = [(query, vector) for query, vector in query_vectors if vector is not None]
        results = []
        for sentence_id, sentence in enumerate(split_sentences(text)):
            sentence_vector = self._vector([token.lower() for token in tokenize_words(sentence)])
            if sentence_vector is None:
                continue
            matches = [(self._cosine(sentence_vector, vector), query) for query, vector in query_vectors]
            if not matches:
                continue
            score, query = max(matches)
            if self.threshold is None or score >= self.threshold:
                results.append(CandidateSentence(sentence_id, sentence, score, query))
        return results