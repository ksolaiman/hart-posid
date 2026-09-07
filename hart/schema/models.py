from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PersonAnnotation:
    document_id: str
    person_id: str
    role: str | None
    attributes: dict[str, Any]


AnnotationIndex = dict[str, list[PersonAnnotation]]
