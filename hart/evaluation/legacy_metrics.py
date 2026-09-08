from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from hart.schema import PersonAnnotation, PersonPrediction


@dataclass(frozen=True)
class MetricResult:
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    false_negative: int


def _prf(true_positive: int, false_positive: int, false_negative: int) -> MetricResult:
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return MetricResult(precision, recall, f1, true_positive, false_positive, false_negative)


def _text(value: Any) -> str | None:
    if value is None:
        return None
    return " ".join(str(value).strip().casefold().split())


def _clothing(attributes: dict[str, Any]) -> list[dict[str, Any]]:
    value = attributes.get("wearing", [])
    if isinstance(value, dict):
        return [
            {"item": item, "descriptors": descriptors}
            for item, descriptors in value.items()
        ]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict) and "item" in item]


def _descriptors(item: dict[str, Any]) -> frozenset[str]:
    value = item.get("descriptors", [])
    if isinstance(value, (str, int, float)):
        value = [value]
    return frozenset(filter(None, (_text(entry) for entry in value)))


def match_people(
    gold: Iterable[PersonAnnotation],
    predictions: Iterable[PersonPrediction],
) -> tuple[list[tuple[PersonAnnotation, PersonPrediction]], int]:
    """Pair people deterministically and return the number of ambiguous matches."""
    gold_people = list(gold)
    predicted_people = list(predictions)
    remaining = list(range(len(predicted_people)))
    pairs: list[tuple[PersonAnnotation, PersonPrediction]] = []
    ambiguous = 0

    for gold_person in gold_people:
        exact = [
            index
            for index in remaining
            if gold_person.person_id == predicted_people[index].person_id
        ]
        if exact:
            index = exact[0]
        else:
            role_matches = [
                index
                for index in remaining
                if gold_person.role
                and predicted_people[index].role
                and _text(gold_person.role) == _text(predicted_people[index].role)
            ]
            if role_matches:
                index = role_matches[0]
            elif len(gold_people) == 1 and len(predicted_people) == 1:
                index = remaining[0]
            else:
                scores = [
                    (_overlap_score(gold_person.attributes, predicted_people[item].attributes), item)
                    for item in remaining
                ]
                scores.sort(reverse=True)
                if not scores:
                    continue
                index = scores[0][1]
                if len(scores) > 1 and scores[0][0] == scores[1][0]:
                    ambiguous += 1
        remaining.remove(index)
        pairs.append((gold_person, predicted_people[index]))
    return pairs, ambiguous


def _overlap_score(gold: dict[str, Any], predicted: dict[str, Any]) -> int:
    score = 0
    for key in ("gender", "race", "height"):
        if _text(gold.get(key)) and _text(gold.get(key)) == _text(predicted.get(key)):
            score += 1
    gold_items = {_text(item.get("item")) for item in _clothing(gold)}
    predicted_items = {_text(item.get("item")) for item in _clothing(predicted)}
    return score + len(gold_items & predicted_items)


def clothing_metrics(
    gold_people: Iterable[PersonAnnotation],
    predicted_people: Iterable[PersonPrediction],
) -> dict[str, MetricResult]:
    pairs, _ = match_people(gold_people, predicted_people)
    attr_only_tp = attr_only_fp = attr_only_fn = 0
    attr_value_tp = attr_value_fp = attr_value_fn = 0

    for gold_person, predicted_person in pairs:
        gold_items = _clothing(gold_person.attributes)
        predicted_items = _clothing(predicted_person.attributes)
        remaining_gold = list(gold_items)
        for predicted_item in predicted_items:
            match_index = next(
                (index for index, gold_item in enumerate(remaining_gold)
                 if _text(gold_item.get("item")) == _text(predicted_item.get("item"))),
                None,
            )
            if match_index is None:
                attr_only_fp += 1
                attr_value_fp += 1
                continue
            gold_item = remaining_gold.pop(match_index)
            attr_only_tp += 1
            if _descriptors(gold_item) == _descriptors(predicted_item):
                attr_value_tp += 1
            else:
                attr_value_fp += 1
                attr_value_fn += 1
        attr_only_fn += len(remaining_gold)
        attr_value_fn += len(remaining_gold)

    return {
        "attr_only": _prf(attr_only_tp, attr_only_fp, attr_only_fn),
        "attr_value": _prf(attr_value_tp, attr_value_fp, attr_value_fn),
    }


def compare_attribute(
    gold_people: Iterable[PersonAnnotation],
    predicted_people: Iterable[PersonPrediction],
    attribute: str,
) -> MetricResult:
    pairs, _ = match_people(gold_people, predicted_people)
    true_positive = false_positive = false_negative = 0
    for gold_person, predicted_person in pairs:
        gold_value = _text(gold_person.attributes.get(attribute))
        predicted_value = _text(predicted_person.attributes.get(attribute))
        if gold_value is not None and predicted_value is not None:
            if gold_value == predicted_value:
                true_positive += 1
            else:
                false_positive += 1
                false_negative += 1
        elif gold_value is not None:
            false_negative += 1
        elif predicted_value is not None:
            false_positive += 1
    return _prf(true_positive, false_positive, false_negative)