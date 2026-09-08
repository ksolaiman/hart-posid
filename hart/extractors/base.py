from __future__ import annotations

from abc import ABC, abstractmethod

from hart.schema import PersonPrediction

PredictionIndex = dict[str, list[PersonPrediction]]


class Extractor(ABC):
    """Small common interface for person-level extraction methods."""

    @abstractmethod
    def extract(self, text: str, document_id: str = "document") -> PredictionIndex:
        raise NotImplementedError


BaseExtractor = Extractor