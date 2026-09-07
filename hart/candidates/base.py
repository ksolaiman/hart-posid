# src/hart/candidates/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class CandidateSentence:
    """
    A sentence selected as potentially containing
    human-attribute information.
    """

    sentence_id: int
    text: str
    score: float
    query: str | None = None


def split_sentences(text: str) -> list[str]:
    """
    Lightweight sentence splitter.

    For the first implementation we avoid tying the candidate-selection
    layer to NLTK or spaCy. This can later be replaced by the tokenizer
    used in the original POSID implementation.
    """
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


class CandidateSelector(ABC):
    """
    Interface for all POSID candidate-sentence extraction methods.

    Given a document and one or more query phrases Q_H,
    return the subset C_s judged relevant.
    """

    @abstractmethod
    def select(
        self,
        text: str,
        queries: Iterable[str],
    ) -> list[CandidateSentence]:
        raise NotImplementedError