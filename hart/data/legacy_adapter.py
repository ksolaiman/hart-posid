from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hart.schema.models import AnnotationIndex, PersonAnnotation
from hart.schema.normalize import normalize_legacy_value, normalize_legacy_wearing


class LegacyAnnotationLoader:
    """Load Schema 1.0 annotations into the canonical runtime model."""

    def load(self, annotation_path: str | Path) -> AnnotationIndex:
        path = Path(annotation_path)
        with path.open(encoding="utf-8") as stream:
            raw_annotations = json.load(stream)

        if not isinstance(raw_annotations, dict):
            raise ValueError("Legacy annotations must be a document-id object")

        annotations: AnnotationIndex = {}
        for document_id, raw_people in raw_annotations.items():
            if not isinstance(document_id, str) or not isinstance(raw_people, list):
                raise ValueError("Legacy annotations must map document IDs to person lists")

            people: list[PersonAnnotation] = []
            for index, raw_person in enumerate(raw_people):
                if not isinstance(raw_person, Mapping):
                    raise ValueError(f"Person annotation {document_id}:{index} must be an object")

                attributes: dict[str, Any] = {}
                for field, value in raw_person.items():
                    if field in {"id", "person_id", "role"}:
                        continue
                    if field == "wearing":
                        attributes[field] = normalize_legacy_wearing(value)
                    else:
                        attributes[field] = normalize_legacy_value(value)

                person_id = normalize_legacy_value(raw_person.get("person_id"))
                if person_id is None:
                    person_id = f"{document_id}:{index}"
                if not isinstance(person_id, str):
                    raise ValueError(f"Person id for {document_id}:{index} must be a string")

                role = normalize_legacy_value(raw_person.get("role"))
                if role is not None and not isinstance(role, str):
                    raise ValueError(f"Role for {document_id}:{index} must be a string")

                people.append(
                    PersonAnnotation(
                        document_id=document_id,
                        person_id=person_id,
                        role=role,
                        attributes=attributes,
                    )
                )

            annotations[document_id] = people

        return annotations
