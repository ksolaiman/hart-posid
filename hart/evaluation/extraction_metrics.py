from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Sequence

from hart.schema import PersonAnnotation, PersonPrediction


@dataclass(frozen=True)
class ExtractionMetricResult:
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    false_negative: int


def _prf(true_positive: int, false_positive: int, false_negative: int) -> ExtractionMetricResult:
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ExtractionMetricResult(
        precision,
        recall,
        f1,
        true_positive,
        false_positive,
        false_negative,
    )


def _normalize(value: Any) -> str | None:
    if value is None:
        return None
    return " ".join(str(value).strip().casefold().split())


def _finite_values(
    people: Iterable[PersonAnnotation | PersonPrediction],
    attribute_name: str,
) -> list[str]:
    return [
        normalized
        for person in people
        if (normalized := _normalize(person.attributes.get(attribute_name))) is not None
    ]


def finite_attribute_metrics(
    gold_people: Iterable[PersonAnnotation],
    predicted_people: Iterable[PersonPrediction],
    attribute_name: str,
) -> ExtractionMetricResult:
    """Evaluate pooled finite values with exact one-to-one multiset matching."""
    gold_values = Counter(_finite_values(gold_people, attribute_name))
    predicted_values = Counter(_finite_values(predicted_people, attribute_name))
    true_positive = sum((gold_values & predicted_values).values())
    false_positive = sum((predicted_values - gold_values).values())
    false_negative = sum((gold_values - predicted_values).values())
    return _prf(true_positive, false_positive, false_negative)


def clothing_instances(
    people: Iterable[PersonAnnotation | PersonPrediction],
) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for person in people:
        wearing = person.attributes.get("wearing", [])
        if isinstance(wearing, dict):
            wearing = [
                {"item": item, "descriptors": descriptors}
                for item, descriptors in wearing.items()
            ]
        if isinstance(wearing, list):
            instances.extend(
                item for item in wearing
                if isinstance(item, dict) and "item" in item
            )
    return instances


def _values(item: dict[str, Any]) -> list[str]:
    raw = item.get("descriptors", [])
    if isinstance(raw, (str, int, float)):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [normalized for value in raw if (normalized := _normalize(value)) is not None]


def _value_overlap(gold: dict[str, Any], predicted: dict[str, Any]) -> int:
    gold_values = Counter(_values(gold))
    predicted_values = Counter(_values(predicted))
    return sum((gold_values & predicted_values).values())


def _maximum_overlap_pairs(
    gold_items: Sequence[dict[str, Any]],
    predicted_items: Sequence[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Match same-name items maximizing descriptor overlap.

    Ties are resolved by the original predicted-item order, then original
    gold-item order. Item-name matching always pairs as many instances as
    possible before maximizing descriptor overlap.
    """
    if not gold_items or not predicted_items:
        return []

    if len(predicted_items) > len(gold_items):
        pairs = _maximum_overlap_pairs(predicted_items, gold_items)
        return [(gold, predicted) for predicted, gold in pairs]

    @lru_cache(maxsize=None)
    def best(predicted_index: int, used_mask: int) -> tuple[int, tuple[int, ...]]:
        if predicted_index == len(predicted_items):
            return 0, ()

        best_score = -1
        best_assignment: tuple[int, ...] = ()
        for gold_index, gold_item in enumerate(gold_items):
            if used_mask & (1 << gold_index):
                continue
            tail_score, tail_assignment = best(
                predicted_index + 1,
                used_mask | (1 << gold_index),
            )
            score = _value_overlap(gold_item, predicted_items[predicted_index]) + tail_score
            assignment = (gold_index,) + tail_assignment
            if score > best_score or (
                score == best_score and assignment < best_assignment
            ):
                best_score = score
                best_assignment = assignment
        return best_score, best_assignment

    _, assignment = best(0, 0)
    return [
        (gold_items[gold_index], predicted_items[predicted_index])
        for predicted_index, gold_index in enumerate(assignment)
    ]


def _group_by_item(items: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        name = _normalize(item.get("item"))
        if name is not None:
            groups.setdefault(name, []).append(item)
    return groups


def evaluate_document(
    gold_people: Iterable[PersonAnnotation],
    predicted_people: Iterable[PersonPrediction],
) -> dict[str, ExtractionMetricResult]:
    gold_groups = _group_by_item(clothing_instances(gold_people))
    predicted_groups = _group_by_item(clothing_instances(predicted_people))
    attr_only_tp = attr_only_fp = attr_only_fn = 0
    attr_value_tp = attr_value_fp = attr_value_fn = 0

    for item_name in sorted(set(gold_groups) | set(predicted_groups)):
        gold_items = gold_groups.get(item_name, [])
        predicted_items = predicted_groups.get(item_name, [])
        pairs = _maximum_overlap_pairs(gold_items, predicted_items)

        attr_only_tp += len(pairs)
        attr_only_fn += len(gold_items) - len(pairs)
        attr_only_fp += len(predicted_items) - len(pairs)

        for gold_item, predicted_item in pairs:
            gold_values = Counter(_values(gold_item))
            predicted_values = Counter(_values(predicted_item))
            value_matches = sum((gold_values & predicted_values).values())
            attr_value_tp += 1 + value_matches
            attr_value_fn += len(gold_values - predicted_values)
            attr_value_fp += len(predicted_values - gold_values)

        matched_gold = {id(gold_item) for gold_item, _ in pairs}
        matched_predicted = {id(predicted_item) for _, predicted_item in pairs}
        for item in gold_items:
            if id(item) not in matched_gold:
                attr_value_fn += 1 + len(_values(item))
        for item in predicted_items:
            if id(item) not in matched_predicted:
                attr_value_fp += 1 + len(_values(item))

    return {
        "attr_only": _prf(attr_only_tp, attr_only_fp, attr_only_fn),
        "attr_value": _prf(attr_value_tp, attr_value_fp, attr_value_fn),
    }


def evaluate_corpus(
    gold: dict[str, list[PersonAnnotation]],
    predictions: dict[str, list[PersonPrediction]],
    document_ids: Iterable[str] | None = None,
) -> dict[str, ExtractionMetricResult]:
    ids = list(document_ids) if document_ids is not None else sorted(gold)
    clothing_per_document = [
        evaluate_document(gold.get(document_id, []), predictions.get(document_id, []))
        for document_id in ids
    ]
    results = {
        metric: _prf(
            sum(result[metric].true_positive for result in clothing_per_document),
            sum(result[metric].false_positive for result in clothing_per_document),
            sum(result[metric].false_negative for result in clothing_per_document),
        )
        for metric in ("attr_only", "attr_value")
    }
    for attribute_name in ("gender", "race", "height"):
        per_document = [
            finite_attribute_metrics(
                gold.get(document_id, []),
                predictions.get(document_id, []),
                attribute_name,
            )
            for document_id in ids
        ]
        results[attribute_name] = _prf(
            sum(result.true_positive for result in per_document),
            sum(result.false_positive for result in per_document),
            sum(result.false_negative for result in per_document),
        )
    return results


def gold_attr_value_units(people: Iterable[PersonAnnotation]) -> int:
    return sum(1 + len(_values(item)) for item in clothing_instances(people))


def corpus_gold_attr_value_units(
    gold: dict[str, list[PersonAnnotation]],
    document_ids: Iterable[str] | None = None,
) -> int:
    ids = list(document_ids) if document_ids is not None else sorted(gold)
    return sum(gold_attr_value_units(gold.get(document_id, [])) for document_id in ids)


def corpus_gold_finite_values(
    gold: dict[str, list[PersonAnnotation]],
    attribute_name: str,
    document_ids: Iterable[str] | None = None,
) -> int:
    ids = list(document_ids) if document_ids is not None else sorted(gold)
    return sum(
        len(_finite_values(gold.get(document_id, []), attribute_name))
        for document_id in ids
    )
