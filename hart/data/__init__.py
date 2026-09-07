from .document import Document, DocumentRecord
from .legacy_adapter import LegacyAnnotationLoader
from .loaders import (
    load_consolidated_metadata,
    load_document,
    load_document_index,
    load_documents,
)

__all__ = [
    "Document",
    "DocumentRecord",
    "LegacyAnnotationLoader",
    "load_consolidated_metadata",
    "load_document",
    "load_document_index",
    "load_documents",
]
