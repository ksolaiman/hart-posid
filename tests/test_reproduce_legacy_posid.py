from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts import reproduce_legacy_posid


AVAILABLE_RESOURCES = {
    "Word2Vec": "AVAILABLE",
    "WordNet": "AVAILABLE",
    "POS tagger": "AVAILABLE",
    "gensim": "AVAILABLE",
    "zero-shot/NLI": "AVAILABLE",
}


def test_parse_methods() -> None:
    args = reproduce_legacy_posid.parse_args(
        ["--methods", "re_posid", "wordnet_posid"]
    )

    assert args.methods == ["re_posid", "wordnet_posid"]


def test_parse_rejects_unknown_method(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        reproduce_legacy_posid.parse_args(["--methods", "foo"])

    error = capsys.readouterr().err
    assert "Unknown method: foo" in error
    assert "Valid methods:" in error
    for method in reproduce_legacy_posid.METHOD_NAMES:
        assert method in error


def test_default_methods_include_all_supported_methods() -> None:
    args = reproduce_legacy_posid.parse_args([])

    assert args.methods == reproduce_legacy_posid.METHOD_NAMES


def test_default_subset_is_full() -> None:
    args = reproduce_legacy_posid.parse_args([])
    records = [_record("synthetic", "Synthetic"), _record("generated", "Generated")]

    assert args.subset == "full"
    assert reproduce_legacy_posid._filter_document_records(records, args.subset) == records


def _record(document_id: str, source_group: str):
    return SimpleNamespace(
        document_id=document_id,
        subset=source_group,
        source=f"{source_group}/{document_id}.txt",
    )


def test_non_generated_subset_selects_expected_28_documents() -> None:
    records = [
        *(_record(f"synthetic-{index}", "Synthetic") for index in range(10)),
        *(_record(f"press-release-{index}", "Press Release") for index in range(13)),
        *(_record(f"incident-report-{index}", "Incident Reports") for index in range(5)),
        _record("generated", "Generated"),
        _record("generated-flash", "Generated_flash"),
        _record("generated-flash-2", "Generated_flash_2"),
    ]

    selected = reproduce_legacy_posid._filter_document_records(
        records, "non_generated_28"
    )

    assert len(selected) == 28
    assert {record.subset for record in selected} == {
        "Synthetic",
        "Press Release",
        "Incident Reports",
    }


@pytest.mark.parametrize(
    ("subset", "expected_source"),
    [
        ("generated_only", "Generated"),
        ("generated_flash_only", "Generated_flash"),
    ],
)
def test_generated_subsets_select_only_requested_source(subset, expected_source) -> None:
    records = [
        _record("generated", "Generated"),
        _record("generated-flash", "Generated_flash"),
        _record("generated-flash-2", "Generated_flash_2"),
        _record("synthetic", "Synthetic"),
    ]

    selected = reproduce_legacy_posid._filter_document_records(records, subset)

    assert [record.document_id for record in selected] == [
        expected_source.lower().replace("_", "-")
    ]


def test_parse_rejects_unknown_subset(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        reproduce_legacy_posid.parse_args(["--subset", "unknown"])

    error = capsys.readouterr().err
    assert "invalid choice: 'unknown'" in error
    for subset in reproduce_legacy_posid.SUBSET_NAMES:
        assert subset in error


def test_non_sbert_methods_do_not_initialize_transformers(monkeypatch) -> None:
    imported_modules = []
    monkeypatch.setattr(
        reproduce_legacy_posid.importlib,
        "import_module",
        lambda name: imported_modules.append(name),
    )

    methods, blocked = reproduce_legacy_posid._build_methods(
        AVAILABLE_RESOURCES,
        ("re_posid", "wordnet_posid", "re_wordnet_posid"),
    )

    assert set(methods) == {"re_posid", "wordnet_posid", "re_wordnet_posid"}
    assert blocked == {}
    assert imported_modules == []


def test_sbert_methods_do_not_initialize_word2vec(monkeypatch) -> None:
    pipeline_calls = []
    classifier = object()

    def pipeline(*args, **kwargs):
        pipeline_calls.append((args, kwargs))
        return classifier

    def import_module(name):
        assert name != "gensim"
        assert name == "transformers"
        return SimpleNamespace(pipeline=pipeline)

    monkeypatch.setattr(reproduce_legacy_posid.importlib, "import_module", import_module)

    methods, blocked = reproduce_legacy_posid._build_methods(
        AVAILABLE_RESOURCES, ("sbert_posid", "re_sbert_posid")
    )

    assert set(methods) == {"sbert_posid", "re_sbert_posid"}
    assert blocked == {}
    assert pipeline_calls == [
        (
            ("zero-shot-classification",),
            {"model": "facebook/bart-large-mnli", "local_files_only": True},
        )
    ]
