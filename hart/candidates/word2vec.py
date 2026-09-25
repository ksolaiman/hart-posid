from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .base import CandidateSelector, CandidateSentence, split_sentences, tokenize_words


class Word2VecSelector(CandidateSelector):
    """Select sentences using token-level Word2Vec similarity."""

    def __init__(self, vectors: Any, threshold: float | None = None):
        self.vectors = vectors.wv if hasattr(vectors, "wv") else vectors
        self.threshold = threshold

    def select(self, text: str, queries: Iterable[str]) -> list[CandidateSentence]:
        queries = list(queries)

        results = []
        for sentence_id, sentence in enumerate(split_sentences(text)):
            best_match: tuple[float, str] | None = None

            for token in tokenize_words(sentence):
                for query in queries:
                    try:
                        score = float(self.vectors.similarity(token, query))
                    except KeyError:
                        continue

                    if best_match is None or score > best_match[0]:
                        best_match = (score, query)

            if best_match is None:
                continue

            score, query = best_match
            if self.threshold is None or score >= self.threshold:
                results.append(CandidateSentence(sentence_id, sentence, score, query, context_text=sentence))

        return results
