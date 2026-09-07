# src/hart/candidates/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import Iterable


try:
    from nltk.tokenize import sent_tokenize, word_tokenize
except ModuleNotFoundError:
    sent_tokenize = None
    word_tokenize = None


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
    Split text with NLTK's sentence tokenizer.

    The fallback keeps this low-level helper usable before the optional NLTK
    tokenizer data has been downloaded.
    """
    if not text:
        return []

    try:
        if sent_tokenize is None:
            raise LookupError
        sentences = sent_tokenize(text)
    except LookupError:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def tokenize_words(text: str) -> list[str]:
    """Tokenize a string with NLTK's word tokenizer."""
    if not text:
        return []

    try:
        if word_tokenize is None:
            raise LookupError
        return word_tokenize(text)
    except LookupError:
        return re.findall(r"\w+(?:['’-]\w+)*|[^\w\s]", text, re.UNICODE)


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