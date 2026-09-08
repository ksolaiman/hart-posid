from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hart.data import LegacyAnnotationLoader, load_document_index
from hart.evaluation.legacy_metrics import (
    MetricResult,
    _clothing,
    _descriptors,
    _prf,
    _text,
    clothing_metrics,
    match_people,
)
from hart.schema import PersonAnnotation, PersonPrediction

METHOD_FILES = {
    "re_posid": "re_posid_predictions.json",
    "wordnet_posid": "wordnet_posid_predictions.json",
    "re_wordnet_posid": "re_wordnet_posid_predictions.json",
}


def _load_predictions(path: Path) -> dict[str, list[PersonPrediction]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        document_id: [PersonPrediction(**person) for person in people]
        for document_id, people in raw.items()
    }


def _items(people: Iterable[PersonAnnotation | PersonPrediction]) -> list[dict[str, Any]]:
    return [item for person in people for item in _clothing(person.attributes)]


def _document_clothing_metrics(
    gold_people: list[PersonAnnotation],
    predicted_people: list[PersonPrediction],
) -> dict[str, MetricResult]:
    gold_items = _items(gold_people)
    predicted_items = _items(predicted_people)
    remaining_gold = list(gold_items)
    ao_tp = ao_fp = ao_fn = 0
    av_tp = av_fp = av_fn = 0

    for predicted_item in predicted_items:
        match_index = next(
            (
                index
                for index, gold_item in enumerate(remaining_gold)
                if _text(gold_item.get("item")) == _text(predicted_item.get("item"))
            ),
            None,
        )
        if match_index is None:
            ao_fp += 1
            av_fp += 1
            continue

        gold_item = remaining_gold.pop(match_index)
        ao_tp += 1
        if _descriptors(gold_item) == _descriptors(predicted_item):
            av_tp += 1
        else:
            av_fp += 1
            av_fn += 1

    ao_fn += len(remaining_gold)
    av_fn += len(remaining_gold)
    return {
        "attr_only": _prf(ao_tp, ao_fp, ao_fn),
        "attr_value": _prf(av_tp, av_fp, av_fn),
    }


def _sum_metric(values: Iterable[MetricResult]) -> MetricResult:
    values = list(values)
    return _prf(
        sum(value.true_positive for value in values),
        sum(value.false_positive for value in values),
        sum(value.false_negative for value in values),
    )


def _metric_dict(value: MetricResult) -> dict[str, Any]:
    return {
        "true_positive": value.true_positive,
        "false_positive": value.false_positive,
        "false_negative": value.false_negative,
        "precision": value.precision,
        "recall": value.recall,
        "f1": value.f1,
    }


def _person_metrics(
    gold: dict[str, list[PersonAnnotation]],
    predictions: dict[str, list[PersonPrediction]],
    document_ids: list[str],
) -> tuple[dict[str, MetricResult], dict[str, int]]:
    clothing: list[dict[str, MetricResult]] = []
    unmatched_gold = unmatched_predictions = ambiguous = 0
    for document_id in document_ids:
        gold_people = gold.get(document_id, [])
        predicted_people = predictions.get(document_id, [])
        clothing.append(clothing_metrics(gold_people, predicted_people))
        matches = match_people(gold_people, predicted_people)
        unmatched_gold += len(matches.unmatched_gold)
        unmatched_predictions += len(matches.unmatched_predictions)
        ambiguous += matches.ambiguous
    return {
        "attr_only": _sum_metric(value["attr_only"] for value in clothing),
        "attr_value": _sum_metric(value["attr_value"] for value in clothing),
    }, {
        "unmatched_gold_people": unmatched_gold,
        "unmatched_predicted_people": unmatched_predictions,
        "ambiguous_matches": ambiguous,
    }


def _document_metrics(
    gold: dict[str, list[PersonAnnotation]],
    predictions: dict[str, list[PersonPrediction]],
    document_ids: list[str],
) -> tuple[dict[str, MetricResult], dict[str, dict[str, MetricResult]]]:
    by_document: dict[str, dict[str, MetricResult]] = {}
    for document_id in document_ids:
        by_document[document_id] = _document_clothing_metrics(
            gold.get(document_id, []), predictions.get(document_id, [])
        )
    return {
        "attr_only": _sum_metric(value["attr_only"] for value in by_document.values()),
        "attr_value": _sum_metric(value["attr_value"] for value in by_document.values()),
    }, by_document


def _person_distribution(
    predictions: dict[str, list[PersonPrediction]],
    document_ids: list[str],
    gold: dict[str, list[PersonAnnotation]],
) -> dict[str, Any]:
    counts = [len(predictions.get(document_id, [])) for document_id in document_ids]
    distribution = Counter(
        "4+" if count >= 4 else str(count)
        for count in counts
    )
    single_gold_counts = Counter(
        "4+" if len(predictions.get(document_id, [])) >= 4 else str(len(predictions.get(document_id, [])))
        for document_id in document_ids
        if len(gold.get(document_id, [])) == 1
    )
    containing = [count for count in counts if count > 0]
    return {
        "total_predicted_people": sum(counts),
        "documents_with_predictions": sum(count > 0 for count in counts),
        "distribution": {key: distribution.get(key, 0) for key in ("0", "1", "2", "3", "4+")},
        "mean_predicted_people_per_document": statistics.mean(counts),
        "median_predicted_people_per_document": statistics.median(counts),
        "max_predicted_people_per_document": max(counts, default=0),
        "mean_predicted_people_per_prediction_document": statistics.mean(containing) if containing else 0.0,
        "gold_single_person_distribution": {
            key: single_gold_counts.get(key, 0) for key in ("0", "1", "2", "3", "4+")
        },
    }


def _person_clothing_summary(people: Iterable[PersonAnnotation | PersonPrediction]) -> list[Any]:
    return [
        {
            "person_id": person.person_id,
            "role": person.role,
            "clothing": _clothing(person.attributes),
        }
        for person in people
    ]


def _high_fragmentation_examples(
    gold: dict[str, list[PersonAnnotation]],
    predictions: dict[str, list[PersonPrediction]],
    document_ids: list[str],
    document_by_id: dict[str, dict[str, MetricResult]],
) -> list[dict[str, Any]]:
    ranked = sorted(
        document_ids,
        key=lambda document_id: len(predictions.get(document_id, [])),
        reverse=True,
    )[:10]
    examples = []
    for document_id in ranked:
        person_result = clothing_metrics(gold.get(document_id, []), predictions.get(document_id, []))
        examples.append({
            "document_id": document_id,
            "gold_person_count": len(gold.get(document_id, [])),
            "predicted_person_count": len(predictions.get(document_id, [])),
            "gold_clothing": _items(gold.get(document_id, [])),
            "predicted_clothing_by_person": _person_clothing_summary(predictions.get(document_id, [])),
            "document_level_predicted_clothing": _items(predictions.get(document_id, [])),
            "person_level": {
                "attr_only": _metric_dict(person_result["attr_only"]),
                "attr_value": _metric_dict(person_result["attr_value"]),
            },
            "document_level": {
                "attr_only": _metric_dict(document_by_id[document_id]["attr_only"]),
                "attr_value": _metric_dict(document_by_id[document_id]["attr_value"]),
            },
        })
    return examples


def _single_person_merge_outcomes(
    gold: dict[str, list[PersonAnnotation]],
    predictions: dict[str, list[PersonPrediction]],
    document_ids: list[str],
) -> dict[str, Any]:
    outcomes = Counter()
    considered = 0
    for document_id in document_ids:
        if len(gold.get(document_id, [])) != 1 or len(predictions.get(document_id, [])) <= 1:
            continue
        considered += 1
        person_metrics = clothing_metrics(gold[document_id], predictions[document_id])
        document_metrics = _document_clothing_metrics(gold[document_id], predictions[document_id])
        person_f1 = person_metrics["attr_only"].f1 + person_metrics["attr_value"].f1
        document_f1 = document_metrics["attr_only"].f1 + document_metrics["attr_value"].f1
        if document_f1 > person_f1:
            outcomes["improve"] += 1
        elif document_f1 < person_f1:
            outcomes["worsen"] += 1
        else:
            outcomes["unchanged"] += 1
    return {
        "documents_considered": considered,
        "outcomes": dict(outcomes),
        "percentages": {
            key: (value / considered * 100 if considered else 0.0)
            for key, value in outcomes.items()
        },
    }


def _write_distribution_csv(path: Path, distributions: dict[str, dict[str, Any]]) -> None:
    fields = ["method", "bucket", "all_documents", "gold_single_person_documents"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method, values in distributions.items():
            for bucket in ("0", "1", "2", "3", "4+"):
                writer.writerow({
                    "method": method,
                    "bucket": bucket,
                    "all_documents": values["distribution"][bucket],
                    "gold_single_person_documents": values["gold_single_person_distribution"][bucket],
                })


def _run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.incitext_root).resolve()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = load_document_index(root)
    document_ids = [record.document_id for record in records]
    gold = LegacyAnnotationLoader().load(root / "Annotation" / "consolidated_annotation_file.json")
    gold_clothing_count = sum(len(_clothing(person.attributes)) for people in gold.values() for person in people)
    if len(document_ids) != 558 or gold_clothing_count != 1503:
        raise RuntimeError(
            f"Expected 558 documents and 1503 gold clothing items; found {len(document_ids)} and {gold_clothing_count}"
        )

    summary: dict[str, Any] = {
        "dataset": {
            "label": "Full-558 InciText diagnostic",
            "documents": len(document_ids),
            "gold_people": sum(len(people) for people in gold.values()),
            "gold_clothing_items": gold_clothing_count,
        },
        "methods": {},
    }
    distributions = {}
    high_examples = {}
    for method, filename in METHOD_FILES.items():
        predictions = _load_predictions(Path(args.prediction_dir) / filename)
        person, person_counts = _person_metrics(gold, predictions, document_ids)
        document, document_by_id = _document_metrics(gold, predictions, document_ids)
        distributions[method] = _person_distribution(predictions, document_ids, gold)
        high_examples[method] = _high_fragmentation_examples(gold, predictions, document_ids, document_by_id)
        summary["methods"][method] = {
            "person_level": {key: _metric_dict(value) for key, value in person.items()},
            "document_level": {key: _metric_dict(value) for key, value in document.items()},
            "person_counts": distributions[method],
            "matching": person_counts,
            "recovery": {
                metric: {
                    "recovered_tp": document[metric].true_positive - person[metric].true_positive,
                    "fraction_person_fn_recovered": (
                        (document[metric].true_positive - person[metric].true_positive)
                        / person[metric].false_negative
                        if person[metric].false_negative else 0.0
                    ),
                    "fraction_person_fp_eliminated": (
                        (person[metric].false_positive - document[metric].false_positive)
                        / person[metric].false_positive
                        if person[metric].false_positive else 0.0
                    ),
                }
                for metric in ("attr_only", "attr_value")
            },
            "single_gold_person_merge_diagnostic": _single_person_merge_outcomes(gold, predictions, document_ids),
        }

    summary["high_fragmentation_examples"] = high_examples
    (output / "person_fragmentation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "high_fragmentation_examples.json").write_text(
        json.dumps(high_examples, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_distribution_csv(output / "person_count_distribution.csv", distributions)
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print("PERSON FRAGMENTATION DIAGNOSTIC — FULL 558")
    print("Method | Person AO F1 | Document AO F1 | Delta | Person AV F1 | Document AV F1 | Delta | Predicted People | Unmatched Predicted")
    for method, values in summary["methods"].items():
        person_ao = values["person_level"]["attr_only"]["f1"]
        document_ao = values["document_level"]["attr_only"]["f1"]
        person_av = values["person_level"]["attr_value"]["f1"]
        document_av = values["document_level"]["attr_value"]["f1"]
        print(
            f"{method} | {person_ao:.3f} | {document_ao:.3f} | {document_ao-person_ao:+.3f} | "
            f"{person_av:.3f} | {document_av:.3f} | {document_av-person_av:+.3f} | "
            f"{values['person_counts']['total_predicted_people']} | "
            f"{values['matching']['unmatched_predicted_people']}"
        )
        for metric in ("attr_only", "attr_value"):
            person = values["person_level"][metric]
            document = values["document_level"][metric]
            print(
                f"  {metric}: TP {person['true_positive']} -> {document['true_positive']}; "
                f"FP {person['false_positive']} -> {document['false_positive']}; "
                f"FN {person['false_negative']} -> {document['false_negative']}; "
                f"document TP+FN={document['true_positive'] + document['false_negative']}"
            )
    re_values = summary["methods"]["re_posid"]
    person_f1 = re_values["person_level"]["attr_only"]["f1"]
    document_f1 = re_values["document_level"]["attr_only"]["f1"]
    person_av_f1 = re_values["person_level"]["attr_value"]["f1"]
    document_av_f1 = re_values["document_level"]["attr_value"]["f1"]
    print(f"\nRE + POSID AO F1: {person_f1:.3f} -> {document_f1:.3f}")
    print(f"RE + POSID AV F1: {person_av_f1:.3f} -> {document_av_f1:.3f}")
    if document_f1 - person_f1 >= 0.20 and document_av_f1 - person_av_f1 >= 0.20:
        interpretation = "YES — strong evidence"
    elif document_f1 - person_f1 >= 0.05 or document_av_f1 - person_av_f1 >= 0.05:
        interpretation = "PARTLY — moderate evidence"
    else:
        interpretation = "NO — little evidence"
    print("Interpretation: " + interpretation)
    print("Classification rule: strong requires both AO and AV F1 deltas >= 0.20; moderate requires either delta >= 0.05; otherwise little evidence.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose person fragmentation without changing official evaluation.")
    parser.add_argument("--incitext-root", default=str(SRC_ROOT.parent / "InciText"))
    parser.add_argument("--prediction-dir", default="outputs/legacy_reproduction")
    parser.add_argument("--output-dir", default="outputs/legacy_reproduction/diagnostics")
    summary = _run(parser.parse_args())
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
