from __future__ import annotations

import json
from pathlib import Path


def load_authoritative_metadata(
    incitext_root: Path,
) -> dict:
    metadata_path = (
        incitext_root
        / "Annotation"
        / "consolidated_report_metadata.json"
    )

    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def annotation_files_from_metadata(
    incitext_root: Path,
    metadata: dict,
) -> list[Path]:
    """
    Derive annotation files directly from consolidated report metadata.

    Do not scan InciText recursively.
    """

    annotation_root = metadata.get("annotation_root")
    documents = metadata.get("documents")
    if not isinstance(annotation_root, str) or not isinstance(documents, list):
        raise ValueError("Malformed consolidated report metadata")

    dataset_root = incitext_root.resolve()
    annotation_files: set[Path] = set()

    for record in documents:
        annotation = record.get("annotation")
        if not isinstance(annotation, dict) or not isinstance(annotation.get("file"), str):
            raise ValueError("Document record has malformed annotation metadata")

        relative_path = Path(annotation_root) / annotation["file"]
        annotation_path = (dataset_root / relative_path).resolve()
        try:
            annotation_path.relative_to(dataset_root)
        except ValueError as error:
            raise ValueError(
                f"Annotation path escapes InciText root: {relative_path}"
            ) from error

        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"Metadata-listed annotation file does not exist: {annotation_path}"
            )
        annotation_files.add(annotation_path)

    return sorted(annotation_files)


def build_consolidated_annotations(
    incitext_root: Path,
    output_path: Path | None = None,
) -> Path:
    """
    Build one consolidated annotation file using only annotation
    sources associated with documents listed in the authoritative
    consolidated report metadata.

    Original annotation files are never modified.
    """

    metadata = load_authoritative_metadata(incitext_root)
    documents = metadata.get("documents")
    if not isinstance(documents, list):
        raise ValueError("Malformed consolidated report metadata")
    document_ids = {record["id"] for record in documents}

    annotation_files = annotation_files_from_metadata(
        incitext_root=incitext_root,
        metadata=metadata,
    )

    combined: dict[str, list[dict]] = {}

    for annotation_file in annotation_files:
        with annotation_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for document_id, persons in data.items():

            # Only keep document IDs that belong to the authoritative
            # metadata-defined dataset.
            if document_id not in document_ids:
                continue

            if document_id in combined:
                if combined[document_id] != persons:
                    raise ValueError(
                        "Conflicting duplicate annotation for "
                        f"{document_id} from {annotation_file}"
                    )
                continue

            combined[document_id] = persons

    missing_ids = document_ids - combined.keys()
    if missing_ids:
        raise ValueError(
            "Metadata-listed documents missing annotations: "
            f"{sorted(missing_ids)}"
        )

    if output_path is None:
        output_path = (
            incitext_root
            / "Annotation"
            / "consolidated_annotation_file.json"
        )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    return output_path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    result = build_consolidated_annotations(project_root / "InciText")
    print(result)
