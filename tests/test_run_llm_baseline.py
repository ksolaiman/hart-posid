from __future__ import annotations

import io
import json
from dataclasses import asdict

import pytest

from hart.schema import PersonAnnotation, PersonPrediction
from scripts import run_llm_baseline as runner


def person(**changes):
    return {"attribution": "person", "role": None, "gender": None,
            "race": None, "height": None, "wearing": [], **changes}


def response(people=(), response_id="resp_test"):
    return {"id": response_id, "model": "gpt-5.6-sol", "status": "completed",
            "usage": {"input_tokens": 20, "output_tokens": 30, "total_tokens": 50},
            "output": [{"type": "message", "content": [{"type": "output_text",
                        "text": json.dumps({"people": list(people)})}]}]}


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "InciText"
    annotation = root / "Annotation"
    annotation.mkdir(parents=True)
    (root / "sources").mkdir()
    (root / "sources" / "doc.txt").write_text("A male suspect wore a black jacket.")
    (annotation / "metadata.json").write_text(json.dumps({"source_root": "sources", "documents": [
        {"id": "doc", "subset": "Synthetic", "source": {"path": "doc.txt", "media_type": "text/plain"}},
    ]}))
    (annotation / "gold.json").write_text(json.dumps({"doc": [
        {"gender": "male", "wearing": {"jacket": ["black"]},
         "shared_group_tags": [{"group_id": "GOLD_SECRET", "support_type": "shared_identical",
                                "fields": ["wearing.jacket"]}]},
    ]}))
    return ["--incitext-root", str(root), "--metadata-filename", "metadata.json",
            "--annotations-filename", "gold.json", "--subset", "full",
            "--output-dir", str(tmp_path / "output")]


def test_dry_run_has_no_api_key_requirement_no_gold_loading_or_output(inputs, monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    def forbidden(*args, **kwargs):
        pytest.fail("dry-run invoked API or gold loader")
    monkeypatch.setattr(runner, "call_responses", forbidden)
    monkeypatch.setattr(runner.LegacyAnnotationLoader, "load", forbidden)
    assert runner.run(runner.parse_args(inputs + ["--dry-run"])) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["document_count"] == 1
    assert summary["config"]["model"] == "gpt-5.6-sol"
    assert summary["config"]["reasoning_effort"] == "high"
    assert not (tmp_path / "output").exists()


def test_api_key_only_from_environment(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_KEY", "alternate-not-allowed")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "alternate-not-allowed")
    monkeypatch.setattr(runner, "urlopen", lambda *a, **k: pytest.fail("missing key must not send"))
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        runner.call_responses(runner.request_payload("text", "gpt-5.6-sol", "high"))
    with pytest.raises(SystemExit):
        runner.parse_args(["--api-key", "not-allowed"])


def test_requests_are_independent_strict_and_use_environment_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-environment-key")
    captured = []
    def fake_open(request, **kwargs):
        assert request.full_url == "https://api.openai.com/v1/responses"
        assert request.get_header("Authorization") == "Bearer test-environment-key"
        captured.append(json.loads(request.data))
        return io.BytesIO(json.dumps(response()).encode())
    monkeypatch.setattr(runner, "urlopen", fake_open)
    for text in ("first document", "second document"):
        runner.call_responses(runner.request_payload(text, "gpt-5.6-sol", "high"))
    for payload, text in zip(captured, ("first document", "second document")):
        assert set(payload) == {"model", "reasoning", "store", "instructions", "input", "text"}
        assert payload["input"] == [{"role": "user", "content": text}]
        assert payload["store"] is False
        assert payload["reasoning"] == {"effort": "high"}
        assert payload["text"]["format"]["strict"] is True
        assert payload["text"]["format"]["schema"] == runner.SCHEMA
    assert "first document" not in json.dumps(captured[1])
    assert "resp_test" not in json.dumps(captured[1])


def test_prompt_limits_mixed_role_document_to_suspect_prediction():
    document = (
        "The victim was a woman wearing a blue coat. "
        "The suspect was a man wearing a black jacket."
    )
    payload = runner.request_payload(document, "gpt-5.6-sol", "high")
    instructions = payload["instructions"]
    assert "regardless of role" not in instructions
    assert "suspects, offenders,\nor persons of interest" in instructions
    for excluded_role in ("victims", "witnesses", "callers", "friends", "officers"):
        assert excluded_role in instructions

    extraction = runner.parse_response(response([
        person(role="suspect", gender="man",
               wearing=[{"item": "jacket", "descriptors": ["black"]}]),
    ]))
    predictions = runner.to_predictions("doc", extraction)
    assert len(predictions) == 1
    assert predictions[0].role == "suspect"
    assert predictions[0].attributes == {
        "gender": "man", "race": None, "height": None,
        "wearing": [{"item": "jacket", "descriptors": ["black"]}],
    }


def test_zero_people():
    assert runner.to_predictions("doc", {"people": []}) == []
    assert runner.parse_response(response()) == {"people": []}


def test_multiple_people_and_clothing_descriptors_map_to_current_schema():
    clothing = [{"item": "jacket", "descriptors": ["black", "leather"]},
                {"item": "pants", "descriptors": ["blue"]}]
    extracted = {"people": [person(role="suspect", gender="male", wearing=clothing),
                             person(gender="female", race="white", height="5 feet 8 inches")]}
    predictions = runner.to_predictions("doc", extracted)
    assert len(predictions) == 2
    assert all(isinstance(value, PersonPrediction) for value in predictions)
    assert predictions[0].attributes["wearing"] == clothing
    assert predictions[0].role == "suspect"
    assert predictions[0].attributes["race"] is None
    assert predictions[1].attributes["height"] == "5 feet 8 inches"
    assert len({value.person_id for value in predictions}) == 2
    assert set(asdict(predictions[0])) == {"document_id", "person_id", "role", "attributes", "metadata"}


def test_group_evidence_is_not_forced_onto_person():
    predictions = runner.to_predictions("doc", {"people": [person(gender="male"),
        person(attribution="group_unspecified", wearing=[{"item": "hoodie", "descriptors": ["grey"]}])]})
    assert predictions[0].attributes["wearing"] == []
    assert predictions[1].person_id is None
    assert predictions[1].role is None
    assert predictions[1].metadata["attribution"] == "group_unspecified"


@pytest.mark.parametrize("extraction", [
    {"people": [person(gold="leak")]}, {"people": [person(wearing={"jacket": ["black"]})]},
    {"people": [person(gender=42)]}, {"people": [person(attribution="invented")]},
    {"people": [person(wearing=[{"item": "shirt", "descriptors": "blue"}])]},
])
def test_invalid_structured_output_rejected(extraction):
    with pytest.raises(ValueError):
        runner.to_predictions("doc", extraction)


@pytest.mark.parametrize("bad_response", [
    {"status": "incomplete"},
    {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal"}]}]},
    {"status": "completed", "output": []},
])
def test_incomplete_refused_or_empty_response_is_not_zero_people(bad_response):
    with pytest.raises(ValueError):
        runner.parse_response(bad_response)


def test_run_resume_and_score_reuse_artifacts_without_gold_leak(inputs, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = []
    def fake_call(payload):
        assert "GOLD_SECRET" not in json.dumps(payload)
        calls.append(payload)
        return response([person(gender="male", wearing=[{"item": "jacket", "descriptors": ["black"]}])])
    monkeypatch.setattr(runner, "call_responses", fake_call)
    assert runner.run(runner.parse_args(inputs)) == 0
    output = tmp_path / "output"
    manifest = json.loads((output / "manifest.json").read_text())
    entry = manifest["responses"]["doc"]
    assert entry["response_id"] == "resp_test"
    assert entry["usage"]["total_tokens"] == 50
    assert entry["latency_seconds"] >= 0
    assert manifest["evaluation"]["annotations_sha256"]
    results = json.loads((output / "results.json").read_text())["llm"]
    assert results["primary_extraction"]["attr_only"]["f1"] == 1
    assert results["primary_extraction"]["gender"]["f1"] == 1
    assert results["person_aware_diagnostic"]["attr_value"]["f1"] == 1
    monkeypatch.delenv("OPENAI_API_KEY")
    assert runner.run(runner.parse_args(inputs + ["--resume"])) == 0
    assert runner.run(runner.parse_args(inputs + ["--score"])) == 0
    assert len(calls) == 1
    with pytest.raises(ValueError, match="differ"):
        runner.run(runner.parse_args(inputs + ["--resume", "--model", "another-model"]))
    with pytest.raises(ValueError, match="already contains"):
        runner.run(runner.parse_args(inputs))


def test_limit_is_per_invocation_and_resume_finishes_full_fourteen_document_cohort(
    tmp_path, monkeypatch,
):
    root = tmp_path / "InciText"
    annotation = root / "Annotation"
    sources = root / "sources"
    annotation.mkdir(parents=True)
    sources.mkdir()
    documents = []
    gold = {}
    for index in range(14):
        document_id = f"doc-{index:02d}"
        filename = f"{document_id}.txt"
        (sources / filename).write_text(f"Document {index} has no target attributes.")
        documents.append({"id": document_id, "subset": "Synthetic",
                          "source": {"path": filename, "media_type": "text/plain"}})
        gold[document_id] = []
    (annotation / "metadata.json").write_text(json.dumps(
        {"source_root": "sources", "documents": documents}
    ))
    (annotation / "gold.json").write_text(json.dumps(gold))
    arguments = ["--incitext-root", str(root), "--metadata-filename", "metadata.json",
                 "--annotations-filename", "gold.json", "--subset", "full",
                 "--output-dir", str(tmp_path / "output")]
    calls = []
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(runner, "call_responses", lambda payload: (
        calls.append(payload["input"][0]["content"]) or response()
    ))

    assert runner.run(runner.parse_args(arguments + ["--limit", "1"])) == 0
    manifest_path = tmp_path / "output" / "manifest.json"
    first_manifest = json.loads(manifest_path.read_text())
    first_document_id = documents[0]["id"]
    first_response = first_manifest["responses"][first_document_id]
    assert len(first_manifest["config"]["documents"]) == 14
    assert "limit" not in first_manifest["config"]
    assert list(first_manifest["responses"]) == [first_document_id]
    assert not (tmp_path / "output" / "results.json").exists()
    assert calls == ["Document 0 has no target attributes."]

    # Reproduce the limited manifest written by the pre-fix runner so the
    # existing challenge14_smoke4 checkpoint remains resumable.
    annotation_hash = first_manifest["config"].pop("annotations_sha256")
    first_manifest["config"]["documents"] = first_manifest["config"]["documents"][:1]
    first_manifest["config"]["limit"] = 1
    first_manifest["config"]["driver_sha256"] = "legacy-driver-hash"
    first_manifest["evaluation"] = {
        "annotations_sha256": annotation_hash, "document_count": 1,
    }
    manifest_path.write_text(json.dumps(first_manifest))

    assert runner.run(runner.parse_args(arguments + ["--resume"])) == 0
    final_manifest = json.loads(manifest_path.read_text())
    assert final_manifest["responses"][first_document_id] == first_response
    assert len(final_manifest["config"]["documents"]) == 14
    assert "limit" not in final_manifest["config"]
    assert final_manifest["config"]["annotations_sha256"] == annotation_hash
    assert len(final_manifest["responses"]) == 14
    assert len(calls) == 14
    assert calls.count("Document 0 has no target attributes.") == 1
    assert calls[1:] == [f"Document {index} has no target attributes." for index in range(1, 14)]
    assert final_manifest["evaluation"]["document_count"] == 14


def test_failures_checkpoint_and_do_not_score(inputs, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(runner, "call_responses", lambda payload: {"id": "resp_bad", "status": "incomplete"})
    with pytest.raises(ValueError, match="not completed"):
        runner.run(runner.parse_args(inputs))
    output = tmp_path / "output"
    assert not (output / "results.json").exists()
    assert json.loads((output / "manifest.json").read_text())["responses"]["doc"]["status"] == "failed"
    with pytest.raises(ValueError, match="incomplete coverage"):
        runner.run(runner.parse_args(inputs + ["--score"]))


def test_scores_use_existing_shared_gold_behavior():
    gold = {"doc": [PersonAnnotation("doc", f"doc:{index}", None,
        {"wearing": [{"item": "sweatshirt", "descriptors": ["hooded"]}]},
        {"shared_group_tags": [{"group_id": "shared", "support_type": "shared_identical",
                                "fields": ["wearing.sweatshirt"]}]}) for index in range(5)]}
    predictions = {"doc": runner.to_predictions("doc", {"people": [person(
        attribution="group_unspecified", wearing=[{"item": "sweatshirt", "descriptors": ["hooded"]}])]})}
    result = runner.score_predictions(gold, predictions, ["doc"])["llm"]
    assert result["primary_extraction"]["attr_value"]["f1"] == 1
    assert result["counts"]["gold_attr_value_units"] == 2


@pytest.mark.parametrize("option", ["--metadata-filename", "--annotations-filename"])
def test_filename_paths_rejected(option):
    with pytest.raises(SystemExit):
        runner.parse_args([option, "../other.json"])


def test_limit_and_dry_run_score_validation():
    with pytest.raises(SystemExit):
        runner.parse_args(["--limit", "0"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--dry-run", "--score"])
