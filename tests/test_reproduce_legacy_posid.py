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


def test_data_filenames_default_and_accept_custom_names() -> None:
    defaults = reproduce_legacy_posid.parse_args([])
    custom = reproduce_legacy_posid.parse_args([
        "--metadata-filename", "metadata_extension.json",
        "--annotations-filename", "annotation_additions.json",
    ])

    assert defaults.metadata_filename == "consolidated_report_metadata.json"
    assert defaults.annotations_filename == "consolidated_annotation_file.json"
    assert custom.metadata_filename == "metadata_extension.json"
    assert custom.annotations_filename == "annotation_additions.json"


@pytest.mark.parametrize("option", ["--metadata-filename", "--annotations-filename"])
@pytest.mark.parametrize(
    "value",
    ["../file.json", "nested/file.json", r"nested\file.json", "/tmp/file.json", ".."],
)
def test_data_filename_options_reject_paths(option, value) -> None:
    with pytest.raises(SystemExit):
        reproduce_legacy_posid.parse_args([option, value])


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


@pytest.mark.parametrize("first_only", [False, True])
@pytest.mark.parametrize("stop_pos", [False, True])
def test_semantic_first_match_cli_and_method_propagation(monkeypatch, first_only, stop_pos):
    argv = []
    if first_only:
        argv.append("--candidate-selector-non-RE-first-match-only")
    if stop_pos:
        argv.append("--stop-on-unexpected-pos")
    args = reproduce_legacy_posid.parse_args(argv)
    assert args.candidate_selector_non_re_first_match_only is first_only
    assert args.stop_on_unexpected_pos is stop_pos

    vectors = object()
    classifier = object()
    modules = {
        "gensim": SimpleNamespace(models=SimpleNamespace(KeyedVectors=SimpleNamespace(
            load_word2vec_format=lambda *a, **kw: vectors))),
        "transformers": SimpleNamespace(pipeline=lambda *a, **kw: classifier),
    }
    monkeypatch.setattr(reproduce_legacy_posid.importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(reproduce_legacy_posid, "_find_word2vec_resource", lambda: Path("fake-vectors"))
    methods, blocked = reproduce_legacy_posid._build_methods(
        AVAILABLE_RESOURCES,
        candidate_selector_non_re_first_match_only=args.candidate_selector_non_re_first_match_only,
        stop_on_unexpected_pos=args.stop_on_unexpected_pos,
    )
    assert not blocked
    for name, extractor in methods.items():
        assert extractor.stop_on_unexpected_pos is stop_pos
        selector = extractor.candidate_selector
        if name == "re_posid":
            assert not hasattr(selector, "candidate_selector_non_re_first_match_only")
            continue
        if name in {"re_wordnet_posid", "re_sbert_posid"}:
            selector = selector.fallback
        assert selector.candidate_selector_non_re_first_match_only is first_only


@pytest.mark.parametrize("first_only", [False, True])
def test_run_passes_semantic_first_match_option(monkeypatch, tmp_path, first_only):
    argv = ["--output-dir", str(tmp_path), "--subset", "non_generated_28"]
    if first_only:
        argv.append("--candidate-selector-non-RE-first-match-only")
    args = reproduce_legacy_posid.parse_args(argv)
    monkeypatch.setattr(reproduce_legacy_posid, "resource_status", lambda: AVAILABLE_RESOURCES)
    monkeypatch.setattr(
        reproduce_legacy_posid,
        "load_document_index",
        lambda root, metadata_filename: [],
    )
    monkeypatch.setattr(reproduce_legacy_posid, "_filter_document_records", lambda records, subset: records)
    monkeypatch.setattr(reproduce_legacy_posid.LegacyAnnotationLoader, "load", lambda self, path: {})

    def build(status, methods, **kwargs):
        assert kwargs["candidate_selector_non_re_first_match_only"] is first_only
        assert kwargs["stop_on_unexpected_pos"] is False
        raise RuntimeError("propagation verified")

    monkeypatch.setattr(reproduce_legacy_posid, "_build_methods", build)
    with pytest.raises(RuntimeError, match="propagation verified"):
        reproduce_legacy_posid.run(args)


def test_run_uses_selected_data_filenames(monkeypatch, tmp_path) -> None:
    args = reproduce_legacy_posid.parse_args([
        "--incitext-root", str(tmp_path / "InciText"),
        "--output-dir", str(tmp_path / "output"),
        "--subset", "non_generated_28",
        "--metadata-filename", "metadata_extension.json",
        "--annotations-filename", "annotation_additions.json",
    ])
    calls = {}

    def load_index(root, metadata_filename):
        calls["metadata"] = (root, metadata_filename)
        return []

    def load_annotations(_self, path):
        calls["annotations"] = path
        return {}

    monkeypatch.setattr(reproduce_legacy_posid, "resource_status", lambda: AVAILABLE_RESOURCES)
    monkeypatch.setattr(reproduce_legacy_posid, "load_document_index", load_index)
    monkeypatch.setattr(
        reproduce_legacy_posid,
        "_filter_document_records",
        lambda records, subset: records,
    )
    monkeypatch.setattr(reproduce_legacy_posid.LegacyAnnotationLoader, "load", load_annotations)
    monkeypatch.setattr(reproduce_legacy_posid, "_build_methods", lambda *args, **kwargs: ({}, {}))

    assert reproduce_legacy_posid.run(args) == 0
    incitext_root = (tmp_path / "InciText").resolve()
    assert calls["metadata"] == (incitext_root, "metadata_extension.json")
    assert calls["annotations"] == (
        incitext_root / "Annotation" / "annotation_additions.json"
    )
