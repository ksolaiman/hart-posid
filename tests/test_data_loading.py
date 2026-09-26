from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest

from hart.data import (
    DocumentRecord,
    LegacyAnnotationLoader,
    load_consolidated_metadata,
    load_document,
    load_document_index,
)
from hart.schema import PersonAnnotation


INCITEXT_ROOT = Path(__file__).resolve().parents[2] / "InciText"
METADATA_PATH = INCITEXT_ROOT / "Annotation" / "consolidated_report_metadata.json"
ANNOTATION_PATH = INCITEXT_ROOT / "Annotation" / "consolidated_annotation_file.json"


def test_real_incident_text_data_loads_from_authoritative_metadata():
    with METADATA_PATH.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    records = metadata["documents"]
    authoritative_ids = [record["id"] for record in records]
    authoritative_sources = {
        (Path(metadata["source_root"]) / record["source"]["path"]).as_posix()
        for record in records
    }

    with patch("hart.data.loaders._read_document_text") as read_document_text:
        document_index = load_document_index(INCITEXT_ROOT)
    read_document_text.assert_not_called()

    assert len(records) == 558
    assert len(document_index) == 558
    assert all(isinstance(record, DocumentRecord) for record in document_index)
    assert len({record.document_id for record in document_index}) == len(authoritative_ids)
    assert {record.document_id for record in document_index} == set(authoritative_ids)
    assert {record.source for record in document_index} == authoritative_sources

    loaded_count = 0
    for record in document_index:
        document = load_document(record, INCITEXT_ROOT)
        assert document.document_id == record.document_id
        assert document.source == record.source
        assert isinstance(document.text, str)
        assert (INCITEXT_ROOT / document.source).is_file()
        loaded_count += 1

    assert loaded_count == 558


def test_real_legacy_annotations_load_into_canonical_person_lists():
    with METADATA_PATH.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    with ANNOTATION_PATH.open(encoding="utf-8") as stream:
        raw_annotations = json.load(stream)

    annotations = LegacyAnnotationLoader().load(ANNOTATION_PATH)

    assert isinstance(annotations, dict)
    assert set(annotations) == set(raw_annotations)
    assert all(isinstance(people, list) for people in annotations.values())
    assert all(
        isinstance(person, PersonAnnotation)
        for people in annotations.values()
        for person in people
    )
    assert all(
        not _contains_literal_null(person.attributes)
        for people in annotations.values()
        for person in people
    )

    person_counts = Counter(len(people) for people in annotations.values())
    assert len(annotations) == 558
    assert sum(len(people) for people in annotations.values()) == 553
    assert person_counts == {0: 6, 1: 551, 2: 1}

    multi_person_ids = [
        document_id
        for document_id, people in annotations.items()
        if len(people) == 2
    ]
    assert len(multi_person_ids) == 1
    first_person, second_person = annotations[multi_person_ids[0]]
    assert first_person is not second_person
    assert first_person.person_id != second_person.person_id

    authoritative_ids = {record["id"] for record in metadata["documents"]}
    assert set(annotations) == authoritative_ids


def test_document_index_accepts_metadata_filename_under_annotation(tmp_path):
    incitext_root = tmp_path / "InciText"
    annotation_dir = incitext_root / "Annotation"
    source_dir = incitext_root / "Documents"
    annotation_dir.mkdir(parents=True)
    source_dir.mkdir()
    (source_dir / "example.txt").write_text("Example", encoding="utf-8")
    metadata = {
        "source_root": "Documents",
        "documents": [
            {
                "id": "example",
                "source": {"path": "example.txt", "media_type": "text/plain"},
            }
        ],
    }
    (annotation_dir / "custom_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    assert load_consolidated_metadata(
        incitext_root, "custom_metadata.json"
    ) == metadata
    records = load_document_index(incitext_root, "custom_metadata.json")
    assert [record.document_id for record in records] == ["example"]


@pytest.mark.parametrize(
    "filename",
    ["../metadata.json", "nested/metadata.json", r"nested\metadata.json", "/tmp/metadata.json", ".."],
)
@pytest.mark.parametrize("loader", [load_consolidated_metadata, load_document_index])
def test_metadata_filename_rejects_paths(tmp_path, filename, loader):
    with pytest.raises(ValueError, match="must be a filename"):
        loader(tmp_path, filename)


def _contains_literal_null(value):
    if value == "null":
        return True
    if isinstance(value, dict):
        return any(_contains_literal_null(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_literal_null(item) for item in value)
    return False
