from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hart.data import LegacyAnnotationLoader, load_document, load_document_index
from scripts.prepare_facebook21_metadata import (
    DOCUMENT_COUNT,
    METADATA_FILENAME,
    SOURCE_EXCEL,
    SOURCE_GOLD,
    load_facebook21_inputs,
    prepare,
)
from scripts.run_facebook21_posid import build_command


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_facebook21_excel_and_gold_have_matching_unique_uuids() -> None:
    records, gold = load_facebook21_inputs()
    document_ids = [document_id for document_id, _ in records]

    assert len(records) == DOCUMENT_COUNT
    assert len(set(document_ids)) == DOCUMENT_COUNT
    assert set(document_ids) == set(gold)
    assert all(description.strip() for _, description in records)


def test_prepare_is_loadable_and_does_not_change_source_data(tmp_path: Path) -> None:
    source_hashes = {_sha256(SOURCE_EXCEL), _sha256(SOURCE_GOLD)}
    output_root = tmp_path / "Facebook"
    expected_ids = prepare(output_root)

    records = load_document_index(output_root, METADATA_FILENAME)
    annotations = LegacyAnnotationLoader().load(output_root / "Annotation" / "goldfb21.json")
    assert [record.document_id for record in records] == expected_ids
    assert len(records) == DOCUMENT_COUNT
    assert set(annotations) == set(expected_ids)
    assert all(len(people) == 1 for people in annotations.values())
    assert all(load_document(record, output_root).text for record in records)
    assert {_sha256(SOURCE_EXCEL), _sha256(SOURCE_GOLD)} == source_hashes


def test_continuous_and_early_stop_commands_use_the_same_21_documents(tmp_path: Path) -> None:
    output_root = tmp_path / "Facebook"
    expected_ids = prepare(output_root)
    metadata = json.loads(
        (output_root / "Annotation" / METADATA_FILENAME).read_text(encoding="utf-8")
    )
    metadata_ids = [document["id"] for document in metadata["documents"]]

    continuous = build_command("continuous")
    early_stop = build_command("early_stop")
    assert metadata_ids == expected_ids
    assert len(metadata_ids) == DOCUMENT_COUNT
    assert "--stop-on-unexpected-pos" not in continuous
    continuous_without_output = continuous[:11] + continuous[13:]
    early_stop_without_output = early_stop[:11] + early_stop[13:-1]
    assert early_stop_without_output == continuous_without_output
    assert early_stop[-1] == "--stop-on-unexpected-pos"
