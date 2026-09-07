 # src/hart/candidates/regex.py

from __future__ import annotations

import re
from typing import Iterable

from .base import (
    CandidateSelector,
    CandidateSentence,
    split_sentences,
)


class RegexSelector(CandidateSelector):
    """
    Exact/pattern-based candidate sentence selector.

    This corresponds to the RE candidate-search baseline in POSID.
    """

    def __init__(
        self,
        ignore_case: bool = True,
        word_boundaries: bool = True,
    ):
        self.ignore_case = ignore_case
        self.word_boundaries = word_boundaries

    def _compile_query(self, query: str) -> re.Pattern:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        escaped = re.escape(query)

        if self.word_boundaries:
            pattern = rf"\b{escaped}\b"
        else:
            pattern = escaped

        flags = re.IGNORECASE if self.ignore_case else 0

        return re.compile(pattern, flags)

    def select(
        self,
        text: str,
        queries: Iterable[str],
    ) -> list[CandidateSentence]:

        sentences = split_sentences(text)
        queries = list(queries)

        results: list[CandidateSentence] = []

        for sentence_id, sentence in enumerate(sentences):

            for query in queries:

                pattern = self._compile_query(query)

                if pattern.search(sentence):
                    results.append(
                        CandidateSentence(
                            sentence_id=sentence_id,
                            text=sentence,
                            score=1.0,
                            query=query,
                        )
                    )

                    # One sentence should only appear once,
                    # even if multiple query phrases match.
                    break

        return results