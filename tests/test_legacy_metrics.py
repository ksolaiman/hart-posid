import pytest

from hart.schema.normalize import canonicalize_clothing_item
from hart.evaluation.legacy_metrics import clothing_metrics, compare_attribute, match_people
from hart.schema import PersonAnnotation, PersonPrediction


def _gold(items):
    return [PersonAnnotation("doc", "doc:0", None, {"wearing": items})]


def _prediction(items):
    return [PersonPrediction("doc", "doc:0", None, {"wearing": items})]


def test_clothing_metrics_distinguish_item_and_descriptor_matches():
    gold = _gold([
        {"item": "jacket", "descriptors": ["red"]},
        {"item": "pants", "descriptors": ["blue"]},
    ])
    predicted = _prediction([
        {"item": "jacket", "descriptors": ["green"]},
        {"item": "hat", "descriptors": ["black"]},
    ])

    metrics = clothing_metrics(gold, predicted)
    assert metrics["attr_only"].true_positive == 1
    assert metrics["attr_only"].false_positive == 1
    assert metrics["attr_only"].false_negative == 1
    assert metrics["attr_value"].true_positive == 0
    assert metrics["attr_value"].false_positive == 2
    assert metrics["attr_value"].false_negative == 2


def test_zero_predictions_count_all_gold_clothing_as_false_negatives():
    metrics = clothing_metrics(
        _gold([
            {"item": "shirt", "descriptors": ["red"]},
            {"item": "pants", "descriptors": ["blue"]},
        ]),
        [],
    )

    assert metrics["attr_only"].true_positive == 0
    assert metrics["attr_only"].false_positive == 0
    assert metrics["attr_only"].false_negative == 2


def test_extra_predicted_person_counts_clothing_as_false_positive():
    gold = _gold([{"item": "shirt", "descriptors": ["red"]}])
    predictions = [
        _prediction([{"item": "shirt", "descriptors": ["red"]}])[0],
        PersonPrediction(
            "doc",
            "doc:1",
            None,
            {"wearing": [{"item": "jacket", "descriptors": ["black"]}]},
        ),
    ]

    metrics = clothing_metrics(gold, predictions)

    assert metrics["attr_only"].true_positive == 1
    assert metrics["attr_only"].false_positive == 1
    assert metrics["attr_only"].false_negative == 0


def test_missing_prediction_counts_fixed_attribute_as_false_negative():
    gold = [PersonAnnotation("doc", "doc:0", None, {"gender": "male"})]

    metrics = compare_attribute(gold, [], "gender")

    assert metrics.true_positive == 0
    assert metrics.false_positive == 0
    assert metrics.false_negative == 1


def test_extra_predicted_person_counts_fixed_attribute_as_false_positive():
    prediction = PersonPrediction("doc", "doc:0", None, {"gender": "male"})

    metrics = compare_attribute([], [prediction], "gender")

    assert metrics.true_positive == 0
    assert metrics.false_positive == 1
    assert metrics.false_negative == 0


def test_gold_clothing_denominator_is_constant_across_prediction_sets():
    gold = _gold([
        {"item": "shirt", "descriptors": ["red"]},
        {"item": "pants", "descriptors": ["blue"]},
    ])
    predictions = [
        _prediction([{"item": "shirt", "descriptors": ["red"]}])[0],
        PersonPrediction(
            "doc",
            "doc:1",
            None,
            {"wearing": [{"item": "hat", "descriptors": ["black"]}]},
        ),
    ]

    complete = clothing_metrics(gold, _prediction([
        {"item": "shirt", "descriptors": ["red"]},
        {"item": "pants", "descriptors": ["blue"]},
    ]))
    partial = clothing_metrics(gold, predictions)

    assert complete["attr_only"].true_positive + complete["attr_only"].false_negative == 2
    assert partial["attr_only"].true_positive + partial["attr_only"].false_negative == 2
    assert complete["attr_value"].true_positive + complete["attr_value"].false_negative == 2
    assert partial["attr_value"].true_positive + partial["attr_value"].false_negative == 2


def test_matching_exposes_unmatched_people_without_changing_ambiguity_index():
    result = match_people(_gold([{"item": "shirt", "descriptors": []}]), [])

    assert result[1] == 0
    assert len(result.unmatched_gold) == 1
    assert result.unmatched_predictions == []

@pytest.mark.parametrize("left,right,equal", [
    ("hoodie", "hoodies", True),
    ("t-shirt", "t-shirts", True),
    ("sweatshirt", "sweatshirts", True),
    ("jacket", "jackets", True),
    ("pants", "pants", True),
    ("jeans", "jeans", True),
    ("pants", "pant", False),
    ("jeans", "jean", False),
    ("hoodie", "jacket", False),
    ("clothes", "clothing", False),
    ("  T‑SHIRTS. ", "t shirt", True),
])
def test_clothing_morphology_is_symmetric_and_conservative(left, right, equal):
    assert canonicalize_clothing_item("pants") == "pants"
    assert canonicalize_clothing_item("jeans") == "jeans"
    for gold_name, predicted_name in ((left, right), (right, left)):
        metrics = clothing_metrics(
            _gold([{"item": gold_name, "descriptors": []}]),
            _prediction([{"item": predicted_name, "descriptors": []}]),
        )
        for view in ("attr_only", "attr_value"):
            assert metrics[view].true_positive == int(equal)
            assert metrics[view].false_positive == int(not equal)
            assert metrics[view].false_negative == int(not equal)
