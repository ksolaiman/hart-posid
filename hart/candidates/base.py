# src/hart/candidates/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

from nltk.tokenize import sent_tokenize, word_tokenize


@dataclass(frozen=True)
class CandidateSentence:
    """
    A sentence selected as potentially containing
    human-attribute information.
    """

    sentence_id: int
    text: str                           # method's actual returned candidate
    score: float
    query: str | None = None

    context_text: str | None = None     # full containing sentence/context


def split_sentences(text: str) -> list[str]:
    """
    Split text with NLTK's sentence tokenizer.
    """
    if not text:
        return []

    return [
        sentence.strip()
        for sentence in sent_tokenize(text)
        if sentence.strip()
    ]


def tokenize_words(text: str) -> list[str]:
    """Tokenize a string with NLTK's word tokenizer."""
    if not text:
        return []

    return word_tokenize(text)


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
