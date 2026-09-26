from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .document import Document, DocumentRecord


DEFAULT_METADATA_FILENAME = "consolidated_report_metadata.json"


def _annotation_filename_path(
    incitext_root: str | Path,
    filename: str,
) -> Path:
    if (
        not filename
        or Path(filename).is_absolute()
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
    ):
        raise ValueError(f"Annotation filename must be a filename, not a path: {filename!r}")
    return Path(incitext_root) / "Annotation" / filename


def load_consolidated_metadata(
    incitext_root: str | Path,
    metadata_filename: str = DEFAULT_METADATA_FILENAME,
) -> dict[str, Any]:
    metadata_path = _annotation_filename_path(incitext_root, metadata_filename)
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)

    if not isinstance(metadata, dict) or not isinstance(metadata.get("documents"), list):
        raise ValueError(f"Malformed consolidated metadata: {metadata_path}")
    return metadata


def _resolve_source_path(
    incitext_root: Path,
    source_root: str,
    source_path: str,
) -> tuple[Path, str]:
    relative_path = Path(source_root) / source_path
    if relative_path.is_absolute():
        raise ValueError(f"Metadata source path must be relative: {relative_path}")

    resolved_path = (incitext_root / relative_path).resolve()
    try:
        resolved_path.relative_to(incitext_root)
    except ValueError as error:
        raise ValueError(f"Metadata source path escapes InciText root: {relative_path}") from error

    if not resolved_path.is_file():
        raise FileNotFoundError(f"Metadata source file does not exist: {resolved_path}")
    return resolved_path, relative_path.as_posix()


def _read_document_text(source_path: Path, media_type: str) -> str:
    if media_type == "text/plain":
        try:
            text = source_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = source_path.read_text(encoding="mac_roman")
    elif media_type == "application/pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError("pypdf is required to load PDF source documents") from error
        reader = PdfReader(source_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        raise ValueError(f"Unsupported source media type {media_type!r}: {source_path}")

    return text


def load_document_index(
    incitext_root: str | Path,
    metadata_filename: str = DEFAULT_METADATA_FILENAME,
) -> list[DocumentRecord]:
    root = Path(incitext_root).resolve()
    metadata = load_consolidated_metadata(root, metadata_filename)
    source_root = metadata.get("source_root")
    if not isinstance(source_root, str) or not source_root:
        raise ValueError("Consolidated metadata must define a non-empty source_root")

    records: list[DocumentRecord] = []
    seen_ids: set[str] = set()
    for record in metadata["documents"]:
        if not isinstance(record, dict):
            raise ValueError("Each consolidated metadata document must be an object")

        document_id = record.get("id")
        source = record.get("source")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("Each consolidated metadata document must have a non-empty id")
        if document_id in seen_ids:
            raise ValueError(f"Duplicate document id in consolidated metadata: {document_id}")
        if not isinstance(source, dict):
            raise ValueError(f"Document {document_id} must have source metadata")

        source_path = source.get("path")
        media_type = source.get("media_type")
        if not isinstance(source_path, str) or not isinstance(media_type, str):
            raise ValueError(f"Document {document_id} has malformed source metadata")

        _, canonical_source = _resolve_source_path(root, source_root, source_path)
        is_synthetic = record.get("is_synthetic")
        if is_synthetic is not None and not isinstance(is_synthetic, bool):
            raise ValueError(f"Document {document_id} has a non-boolean is_synthetic value")

        records.append(
            DocumentRecord(
                document_id=document_id,
                source=canonical_source,
                media_type=media_type,
                subset=record.get("subset"),
                is_synthetic=is_synthetic,
                metadata=dict(record),
            )
        )
        seen_ids.add(document_id)

    return records


def load_document(record: DocumentRecord, incitext_root: str | Path) -> Document:
    root = Path(incitext_root).resolve()
    resolved_path, canonical_source = _resolve_source_path(root, "", record.source)
    return Document(
        document_id=record.document_id,
        text=_read_document_text(resolved_path, record.media_type),
        source=canonical_source,
        subset=record.subset,
        is_synthetic=record.is_synthetic,
        metadata=dict(record.metadata),
    )


def load_documents(incitext_root: str | Path) -> Iterator[Document]:
    for record in load_document_index(incitext_root):
        yield load_document(record, incitext_root)
