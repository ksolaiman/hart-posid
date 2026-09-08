from scripts.diagnose_person_fragmentation import _document_clothing_metrics
from hart.schema import PersonAnnotation, PersonPrediction


def _gold(items):
    return [PersonAnnotation("doc", "doc:0", None, {"wearing": items})]


def _prediction(items):
    return [PersonPrediction("doc", "doc:0", None, {"wearing": items})]


def test_document_metrics_preserve_duplicate_clothing_items():
    gold = _gold([
        {"item": "shirt", "descriptors": ["red"]},
        {"item": "shirt", "descriptors": ["blue"]},
    ])
    predicted = _prediction([
        {"item": "shirt", "descriptors": ["red"]},
    ])

    metrics = _document_clothing_metrics(gold, predicted)

    assert metrics["attr_only"].true_positive == 1
    assert metrics["attr_only"].false_negative == 1
    assert metrics["attr_only"].true_positive + metrics["attr_only"].false_negative == 2


def test_document_attr_value_wrong_descriptor_is_fp_and_fn():
    gold = _gold([{"item": "jacket", "descriptors": ["red"]}])
    predicted = _prediction([{"item": "jacket", "descriptors": ["green"]}])

    metrics = _document_clothing_metrics(gold, predicted)

    assert metrics["attr_value"].true_positive == 0
    assert metrics["attr_value"].false_positive == 1
    assert metrics["attr_value"].false_negative == 1
