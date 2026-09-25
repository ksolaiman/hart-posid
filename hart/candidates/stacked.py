from __future__ import annotations

from collections.abc import Iterable

from .base import CandidateSelector, CandidateSentence


class StackedSelector(CandidateSelector):
    """Combine primary and semantic candidates, preserving extraction roles."""

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

    # def select(self, text: str, queries: Iterable[str] = ()) -> list[CandidateSentence]:
    #     common_queries = list(queries)
    #     primary_queries = self.primary_queries if self.primary_queries is not None else common_queries
    #     fallback_queries = self.fallback_queries if self.fallback_queries is not None else common_queries

    #     candidates = self.primary.select(text, primary_queries)

    #     # the following 'if' is returning if regex finds any candidates
    #     # but fallback model can catch candidates that were not caught by primary regex
    #     # need to consolidate both at end. SO COMMENTED out now -
    #     # if candidates:
    #     #     return candidates

    #     if self.fallback_queries is None:
    #         if candidates:
    #             return candidates
    #         return self.fallback.select(text, fallback_queries)     # if none found by primary, return fallbacks's candidates

    #     clothing_primary = [
    #         candidate
    #         for candidate in candidates
    #         if candidate.query in fallback_queries
    #     ]

    #     if clothing_primary:
    #         return candidates

    #     fallback_candidates = self.fallback.select(
    #         text,
    #         fallback_queries,
    #     )

    #     return fallback_candidates + candidates
    
    #     # return self.fallback.select(text, fallback_queries)


    def select(
            self,
            text: str,
            queries: Iterable[str] = (),
        ) -> list[CandidateSentence]:

        common_queries = list(queries)

        primary_queries = (
            self.primary_queries
            if self.primary_queries is not None
            else common_queries
        )
        fallback_queries = (
            self.fallback_queries
            if self.fallback_queries is not None
            else common_queries
        )

        # Regex runs with ["wearing", fixed_query].
        regex_candidates = self.primary.select(text, primary_queries)

        # POSID deduplicates equivalent extraction work by sentence and role.
        # Preserve every candidate here, including distinct roles for one sentence.
        semantic_candidates = self.fallback.select(text, fallback_queries)
        return regex_candidates + semantic_candidates
