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
    ):
        self.ignore_case = ignore_case

    def _compile_query(self, query: str) -> re.Pattern:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        pattern = rf"{re.escape(query)}[^.]*"
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

                match = pattern.search(sentence)

                if match:
                    context_text = sentence
                    if (
                        sentence_id > 0
                        and re.match(r"\s*wear(?:ing|s)?\b", sentence, re.IGNORECASE)
                        and re.search(
                            r"\b(?:suspect|victim|juvenile)\b",
                            sentences[sentence_id - 1],
                            re.IGNORECASE,
                        )
                    ):
                        context_text = f"{sentences[sentence_id - 1]} {sentence}"
                    results.append(
                        CandidateSentence(
                            sentence_id=sentence_id,
                            text=match.group(0).strip(),
                            score=1.0,
                            query=query,
                            context_text=context_text,
                        )
                    )

                    # One sentence should only appear once,
                    # even if multiple query phrases match.
                    break

        return results
