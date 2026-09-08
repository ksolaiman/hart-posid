from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PersonAnnotation:
    document_id: str
    person_id: str
    role: str | None
    attributes: dict[str, Any]


@dataclass(frozen=True)
class PersonPrediction:
    document_id: str
    person_id: str | None
    role: str | None
    attributes: dict[str, Any]
    metadata: dict[str, Any] | None = None


AnnotationIndex = dict[str, list[PersonAnnotation]]
