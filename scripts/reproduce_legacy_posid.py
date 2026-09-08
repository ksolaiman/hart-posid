from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import nltk

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hart.candidates import RegexSelector, StackedSelector, Word2VecSelector, WordNetSelector, ZeroShotSelector
from hart.data import LegacyAnnotationLoader, load_document_index, load_document
from hart.evaluation.legacy_metrics import clothing_metrics, compare_attribute, match_people
from hart.evaluation.extraction_metrics import (
    corpus_gold_attr_value_units,
    evaluate_corpus,
)
from hart.extractors import POSIDExtractor


PUBLISHED = {
    "word2vec_posid": (0.52, 0.49),
    "re_posid": (0.84, 0.87),
    "wordnet_posid": (0.49, 0.45),
    "sbert_posid": (0.62, 0.59),
    "re_wordnet_posid": (0.77, 0.90),
    "re_sbert_posid": (0.87, 0.90),
}
PUBLISHED_TABLE_I = {
    "gender": (0.94, 0.73, 0.82),
    "race": (0.94, 0.73, 0.82),
    "height": (0.72, 0.57, 0.63),
    "clothes_attr_only": (0.87, 0.87, 0.87),
    "clothes_attr_value": (0.92, 0.87, 0.90),
}


def resource_status() -> dict[str, str]:
    status = {}
    for label, resource in {
        "Word2Vec": "models/word2vec_sample/pruned.word2vec.txt",
        "WordNet": "corpora/wordnet",
        "POS tagger": "taggers/averaged_perceptron_tagger_eng/",
    }.items():
        try:
            nltk.data.find(resource)
            status[label] = "AVAILABLE"
        except LookupError:
            try:
                nltk.data.find(resource.rstrip("/") + ".zip")
                status[label] = "AVAILABLE"
            except LookupError:
                status[label] = "MISSING"
    status["gensim"] = "AVAILABLE" if importlib.util.find_spec("gensim") else "MISSING"
    status["zero-shot/NLI"] = (
        "AVAILABLE"
        if importlib.util.find_spec("transformers") and importlib.util.find_spec("torch")
        else "MISSING"
    )
    return status


def _json_prediction(prediction: Any) -> dict[str, Any]:
    return asdict(prediction)


def _find_word2vec_resource() -> Path | None:
    candidates = [Path(nltk.data.find("models/word2vec_sample/pruned.word2vec.txt"))]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _build_methods(status: dict[str, str]) -> tuple[dict[str, Any], dict[str, str]]:
    methods: dict[str, Any] = {}
    blocked: dict[str, str] = {}
    regex = RegexSelector()

    if status["WordNet"] == "AVAILABLE" and status["POS tagger"] == "AVAILABLE":
        wordnet = WordNetSelector(threshold=0.9, synset_strategy="legacy_asymmetric")
        methods["wordnet_posid"] = POSIDExtractor(wordnet, ["clothes"])
        methods["re_wordnet_posid"] = POSIDExtractor(
            StackedSelector(regex, wordnet, ["wearing"], ["clothes"]), []
        )
        methods["re_posid"] = POSIDExtractor(regex, ["wearing"])
    else:
        for name in ("wordnet_posid", "re_wordnet_posid", "re_posid"):
            blocked[name] = "WordNet and/or POS tagger resource is missing"

    if status["Word2Vec"] == "AVAILABLE" and status["gensim"] == "AVAILABLE":
        gensim = importlib.import_module("gensim")
        resource = _find_word2vec_resource()
        if resource is None:
            blocked["word2vec_posid"] = "historical pruned Word2Vec resource is missing"
        else:
            vectors = gensim.models.KeyedVectors.load_word2vec_format(str(resource), binary=False)
            methods["word2vec_posid"] = POSIDExtractor(
                Word2VecSelector(vectors, threshold=0.5), ["clothes"]
            )
    else:
        blocked["word2vec_posid"] = "historical Word2Vec resource or gensim is missing"

    if status["zero-shot/NLI"] == "AVAILABLE":
        pipeline = importlib.import_module("transformers").pipeline
        classifier = pipeline("zero-shot-classification", local_files_only=True)
        zero_shot = ZeroShotSelector(classifier, threshold=0.85)
        methods["sbert_posid"] = POSIDExtractor(zero_shot, ["clothes"])
        methods["re_sbert_posid"] = POSIDExtractor(
            StackedSelector(regex, zero_shot, ["wearing"], ["clothes"]), []
        )
    else:
        blocked["sbert_posid"] = "zero-shot/NLI dependencies or local model are missing"
        blocked["re_sbert_posid"] = "zero-shot/NLI dependencies or local model are missing"
    return methods, blocked


def run(args: argparse.Namespace) -> int:
    incitext_root = Path(args.incitext_root).resolve()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    status = resource_status()
    print("RESOURCE CHECK")
    for name, value in status.items():
        print(f"  {name}: {value}")

    records = load_document_index(incitext_root)
    annotations_path = incitext_root / "Annotation" / "consolidated_annotation_file.json"
    gold = LegacyAnnotationLoader().load(annotations_path)
    document_ids = [record.document_id for record in records]
    gold_attr_value_units = corpus_gold_attr_value_units(gold, document_ids)
    print("DATA CHECK — Full-558 InciText reproduction")
    print(f"  total indexed documents = {len(records)}")
    print(f"  gold person annotations = {sum(len(people) for people in gold.values())}")
    print(f"  zero-person documents = {sum(not people for people in gold.values())}")
    print(f"  single-person documents = {sum(len(people) == 1 for people in gold.values())}")
    print(f"  multi-person documents = {sum(len(people) > 1 for people in gold.values())}")
    if len(records) != 558:
        raise RuntimeError(f"Expected 558 indexed documents, found {len(records)}")

    methods, blocked = _build_methods(status)
    for name, reason in blocked.items():
        print(f"METHOD {name}: BLOCKED — {reason}")

    documents = [load_document(record, incitext_root) for record in records]
    results = {}
    for name, extractor in methods.items():
        predictions: dict[str, list[Any]] = {}
        errors = []
        for document in documents:
            try:
                predictions.update(extractor.extract(document.text, document.document_id))
            except Exception as error:  # report every document failure, do not omit it
                errors.append({"document_id": document.document_id, "error": repr(error)})
                predictions[document.document_id] = []
        path = output_root / f"{name}_predictions.json"
        path.write_text(json.dumps({doc_id: [
            _json_prediction(person) for person in people
        ] for doc_id, people in predictions.items()}, indent=2, sort_keys=True), encoding="utf-8")
        clothing = [clothing_metrics(gold.get(doc.document_id, []), predictions.get(doc.document_id, [])) for doc in documents]
        ao = _sum_metrics(clothing, "attr_only")
        av = _sum_metrics(clothing, "attr_value")
        primary = evaluate_corpus(gold, predictions, document_ids)
        attribute_metrics = {
            attribute: asdict(_sum_attribute_metrics(
                [compare_attribute(gold.get(doc.document_id, []), predictions.get(doc.document_id, []), attribute) for doc in documents]
            ))
            for attribute in ("gender", "race", "height")
        }
        ambiguous_documents = sum(
            bool(match_people(gold.get(doc.document_id, []), predictions.get(doc.document_id, []))[1])
            for doc in documents
        )
        results[name] = {
            # Backward-compatible person-aware fields retained for diagnostics.
            "attr_only": asdict(ao),
            "attr_value": asdict(av),
            "primary_extraction": {
                "attr_only": asdict(primary["attr_only"]),
                "attr_value": asdict(primary["attr_value"]),
            },
            "person_aware_diagnostic": {
                "attr_only": asdict(ao),
                "attr_value": asdict(av),
            },
            "attributes": attribute_metrics,
            "errors": errors,
            "counts": {
                "documents_attempted": len(documents),
                "documents_successfully_processed": len(documents) - len(errors),
                "documents_with_predictions": sum(bool(predictions.get(doc.document_id)) for doc in documents),
                "predicted_people": sum(len(predictions.get(doc.document_id, [])) for doc in documents),
                "evaluation_clothing_items": sum(len(person.attributes.get("wearing", [])) for people in predictions.values() for person in people),
                "gold_clothing_items": sum(
                    len(person.attributes.get("wearing", []))
                    for people in gold.values()
                    for person in people
                ),
                "gold_attr_value_units": gold_attr_value_units,
                "exceptions": len(errors),
                "ambiguous_multi_person_documents": ambiguous_documents,
            },
        }
        print(f"METHOD {name}: PASS — predictions saved to {path}")
        print(f"  documents attempted={len(documents)} successfully processed={len(documents)-len(errors)} documents with predictions={sum(bool(predictions.get(doc.document_id)) for doc in documents)} predicted people={sum(len(predictions.get(doc.document_id, [])) for doc in documents)}")
        print(f"  evaluation clothing items={sum(len(person.attributes.get('wearing', [])) for people in predictions.values() for person in people)} exceptions={len(errors)}")

    (output_root / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    (output_root / "resource_status.json").write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    (output_root / "blocked_methods.json").write_text(json.dumps(blocked, indent=2, sort_keys=True), encoding="utf-8")
    _print_tables(results)
    return 0


def _sum_metrics(metrics: list[dict[str, Any]], key: str):
    from hart.evaluation.legacy_metrics import MetricResult
    tp = sum(item[key].true_positive for item in metrics)
    fp = sum(item[key].false_positive for item in metrics)
    fn = sum(item[key].false_negative for item in metrics)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return MetricResult(precision, recall, f1, tp, fp, fn)


def _sum_attribute_metrics(metrics: list[Any]):
    from hart.evaluation.legacy_metrics import MetricResult
    tp = sum(item.true_positive for item in metrics)
    fp = sum(item.false_positive for item in metrics)
    fn = sum(item.false_negative for item in metrics)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return MetricResult(precision, recall, f1, tp, fp, fn)


def _print_tables(results: dict[str, Any]) -> None:
    print("\nLEGACY CLOTHING COMPARISON — FULL 558 DOCUMENTS")
    print("Method | Published F1 AO/AV | Reproduced F1 AO/AV | Delta AO/AV")
    labels = [
        ("word2vec_posid", "Word2Vec + POSID"), ("re_posid", "RE + POSID"),
        ("wordnet_posid", "WordNet + POSID"), ("sbert_posid", "SBERT + POSID"),
        ("re_wordnet_posid", "RE + WordNet + POSID"), ("re_sbert_posid", "RE + SBERT + POSID"),
    ]
    for key, label in labels:
        published = PUBLISHED[key]
        if key not in results:
            print(f"{label} | {published[0]:.2f}/{published[1]:.2f} | BLOCKED | BLOCKED")
            continue
        reproduced = (
            results[key]["primary_extraction"]["attr_only"]["f1"],
            results[key]["primary_extraction"]["attr_value"]["f1"],
        )
        print(f"{label} | {published[0]:.2f}/{published[1]:.2f} | {reproduced[0]:.2f}/{reproduced[1]:.2f} | {reproduced[0]-published[0]:+.2f}/{reproduced[1]-published[1]:+.2f}")
        primary = results[key]["primary_extraction"]
        diagnostic = results[key]["person_aware_diagnostic"]
        print(f"  PRIMARY P/R/F1 AO={primary['attr_only']['precision']:.2f}/{primary['attr_only']['recall']:.2f}/{primary['attr_only']['f1']:.2f} AV={primary['attr_value']['precision']:.2f}/{primary['attr_value']['recall']:.2f}/{primary['attr_value']['f1']:.2f}")
        print(f"  PERSON-AWARE P/R/F1 AO={diagnostic['attr_only']['precision']:.2f}/{diagnostic['attr_only']['recall']:.2f}/{diagnostic['attr_only']['f1']:.2f} AV={diagnostic['attr_value']['precision']:.2f}/{diagnostic['attr_value']['recall']:.2f}/{diagnostic['attr_value']['f1']:.2f}")

    print("\nTABLE I COMPARISON — RE + SBERT / ZERO-SHOT + POSID — FULL 558")
    print("Attribute | Published P/R/F1 | Reproduced P/R/F1 | Delta")
    if "re_sbert_posid" not in results:
        for attribute, published in PUBLISHED_TABLE_I.items():
            print(f"{attribute} | {published[0]:.2f}/{published[1]:.2f}/{published[2]:.2f} | BLOCKED | BLOCKED")
        return
    best = results["re_sbert_posid"]
    reproduced = {
        "gender": best["attributes"]["gender"],
        "race": best["attributes"]["race"],
        "height": best["attributes"]["height"],
        "clothes_attr_only": best["attr_only"],
        "clothes_attr_value": best["attr_value"],
    }
    for attribute, published in PUBLISHED_TABLE_I.items():
        metric = reproduced[attribute]
        values = (metric["precision"], metric["recall"], metric["f1"])
        delta = tuple(values[index] - published[index] for index in range(3))
        print(f"{attribute} | {published[0]:.2f}/{published[1]:.2f}/{published[2]:.2f} | {values[0]:.2f}/{values[1]:.2f}/{values[2]:.2f} | {delta[0]:+.2f}/{delta[1]:+.2f}/{delta[2]:+.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full-558 InciText legacy POSID reproduction.")
    parser.add_argument("--incitext-root", default=str(SRC_ROOT.parent / "InciText"))
    parser.add_argument("--output-dir", default="outputs/legacy_reproduction")
    raise SystemExit(run(parser.parse_args()))