"""Convert the 14 extension records to the document-index metadata schema."""

import json
from pathlib import Path


def main() -> None:
    incitext_root = (Path(__file__).resolve().parents[2] / "InciText").resolve()
    annotation_dir = incitext_root / "Annotation"
    additions = json.loads((annotation_dir / "report_metadata_additions.json").read_text())
    consolidated = json.loads((annotation_dir / "consolidated_report_metadata.json").read_text())
    source_root = consolidated["source_root"]

    if len(additions) != 14:
        raise ValueError(f"Expected 14 records, found {len(additions)}")
    ids = [record["id"] for record in additions]
    if len(set(ids)) != 14:
        raise ValueError("Extension document IDs must be unique")

    documents = []
    for record in additions:
        source = (incitext_root / source_root / record["directory"]).resolve()
        if not source.is_relative_to(incitext_root) or not source.is_file():
            raise ValueError(f"Source file is missing or outside InciText: {record['directory']}")
        documents.append({
            "id": record["id"],
            "title": record["title"],
            "subset": record["type"],
            "source": {"path": record["directory"], "media_type": "text/plain"},
        })

    output = annotation_dir / "report_metadata_new14.json"
    output.write_text(json.dumps({"source_root": source_root, "documents": documents}, indent=2) + "\n")
    print(f"Output: {output.name}")
    print(f"Documents: {len(documents)}")
    print(f"source_root: {source_root}")


if __name__ == "__main__":
    main()
