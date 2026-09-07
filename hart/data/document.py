from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    source: str
    subset: str | None
    is_synthetic: bool | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    source: str
    media_type: str
    subset: str | None
    is_synthetic: bool | None
    metadata: dict[str, Any]
