from __future__ import annotations

from collections.abc import Iterable

from .base import CandidateSelector, CandidateSentence


class StackedSelector(CandidateSelector):
    """Run a primary selector and use a fallback only when it misses."""

    def __init__(
        self,
        primary: CandidateSelector,
        fallback: CandidateSelector,
        primary_queries: Iterable[str] | None = None,
        fallback_queries: Iterable[str] | None = None,
    ):
        self.primary = primary
        self.fallback = fallback
        self.primary_queries = list(primary_queries) if primary_queries is not None else None
        self.fallback_queries = list(fallback_queries) if fallback_queries is not None else None

    def select(self, text: str, queries: Iterable[str] = ()) -> list[CandidateSentence]:
        common_queries = list(queries)
        primary_queries = self.primary_queries if self.primary_queries is not None else common_queries
        fallback_queries = self.fallback_queries if self.fallback_queries is not None else common_queries

        candidates = self.primary.select(text, primary_queries)
        if candidates:
            return candidates
        return self.fallback.select(text, fallback_queries)