from hart.evaluation.extraction_metrics import evaluate_document
from hart.schema import PersonAnnotation, PersonPrediction


def gold(items, person_id="doc:0"):
    return [PersonAnnotation("doc", person_id, None, {"wearing": items})]


def prediction(items, person_id="doc:0"):
    return [PersonPrediction("doc", person_id, None, {"wearing": items})]


def item(name, descriptors):
    return {"item": name, "descriptors": descriptors}


def test_fragmented_predictions_are_pooled_for_primary_metrics():
    result = evaluate_document(
        gold([item("jacket", ["black"]), item("pants", ["blue"])]),
        prediction([item("jacket", ["black"])])
        + prediction([item("pants", ["blue"])], "doc:1"),
    )

    assert result["attr_only"].true_positive == 2
    assert result["attr_only"].false_positive == 0
    assert result["attr_only"].false_negative == 0
    assert result["attr_value"].true_positive == 4
    assert result["attr_value"].false_positive == 0
    assert result["attr_value"].false_negative == 0


def test_extra_fragment_contributes_item_and_value_false_positives():
    result = evaluate_document(
        gold([item("jacket", ["black"]), item("pants", ["blue"])]),
        prediction([item("jacket", ["black"])])
        + prediction([item("pants", ["blue"]), item("hoodie", ["red"])], "doc:1"),
    )

    assert result["attr_only"].false_positive == 1
    assert result["attr_value"].false_positive == 2


def test_attr_value_allows_partial_descriptor_credit():
    result = evaluate_document(
        gold([item("shirt", ["red", "blue", "buttoned up"])]),
        prediction([item("shirt", ["red", "blue"])]),
    )

    assert result["attr_value"].true_positive == 3
    assert result["attr_value"].false_positive == 0
    assert result["attr_value"].false_negative == 1


def test_duplicate_items_from_multiple_people_are_preserved():
    result = evaluate_document(
        gold([item("shirt", ["blue"]), item("shirt", ["red"])], "doc:0")
        + [PersonAnnotation("doc", "doc:1", None, {"wearing": []})],
        prediction([item("shirt", ["blue"]), item("shirt", ["red"])]),
    )

    assert result["attr_only"].true_positive == 2
    assert result["attr_value"].true_positive == 4


def test_duplicate_item_matching_maximizes_descriptor_overlap():
    result = evaluate_document(
        gold([item("shirt", ["blue"]), item("shirt", ["red"])]),
        prediction([item("shirt", ["red"]), item("shirt", ["blue"])]),
    )

    assert result["attr_value"].true_positive == 4
    assert result["attr_value"].false_positive == 0
    assert result["attr_value"].false_negative == 0


def test_missing_item_counts_item_and_values_as_false_negatives():
    result = evaluate_document(
        gold([item("jacket", ["black", "leather"])]),
        [],
    )

    assert result["attr_value"].true_positive == 0
    assert result["attr_value"].false_positive == 0
    assert result["attr_value"].false_negative == 3


def test_extra_item_counts_item_and_values_as_false_positives():
    result = evaluate_document(
        [],
        prediction([item("jacket", ["black", "leather"])]),
    )

    assert result["attr_value"].true_positive == 0
    assert result["attr_value"].false_positive == 3
    assert result["attr_value"].false_negative == 0


def test_multiplicity_is_not_collapsed():
    result = evaluate_document(
        gold([item("shirt", ["blue"]), item("shirt", ["blue"])]),
        prediction([item("shirt", ["blue"])]),
    )

    assert result["attr_only"].true_positive == 1
    assert result["attr_only"].false_negative == 1
    assert result["attr_value"].true_positive == 2
    assert result["attr_value"].false_negative == 2
