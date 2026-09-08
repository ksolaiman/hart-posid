from hart.evaluation.legacy_metrics import clothing_metrics
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