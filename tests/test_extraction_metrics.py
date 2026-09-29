import pytest

from hart.schema.normalize import canonicalize_clothing_item
from hart.evaluation.extraction_metrics import (
    corpus_gold_finite_values,
    evaluate_corpus,
    evaluate_document,
    finite_attribute_metrics,
)
from hart.schema import PersonAnnotation, PersonPrediction
from hart.schema.normalize import normalize_height_surface


def gold(items, person_id="doc:0"):
    return [PersonAnnotation("doc", person_id, None, {"wearing": items})]


def prediction(items, person_id="doc:0"):
    return [PersonPrediction("doc", person_id, None, {"wearing": items})]


def item(name, descriptors):
    return {"item": name, "descriptors": descriptors}


def gold_attribute(attribute, value, person_id="doc:0"):
    return [PersonAnnotation("doc", person_id, None, {attribute: value})]


def predicted_attribute(attribute, value, person_id="doc:0"):
    return [PersonPrediction("doc", person_id, None, {attribute: value})]


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


def test_finite_attribute_correct_value_is_true_positive():
    result = finite_attribute_metrics(
        gold_attribute("gender", "male"),
        predicted_attribute("gender", "male"),
        "gender",
    )

    assert (result.true_positive, result.false_positive, result.false_negative) == (
        1,
        0,
        0,
    )


def test_finite_attribute_wrong_value_is_false_positive_and_false_negative():
    result = finite_attribute_metrics(
        gold_attribute("gender", "male"),
        predicted_attribute("gender", "female"),
        "gender",
    )

    assert (result.true_positive, result.false_positive, result.false_negative) == (
        0,
        1,
        1,
    )


def test_fragmented_correct_finite_value_and_empty_prediction_are_pooled():
    predictions = predicted_attribute("race", "asian") + predicted_attribute(
        "race", None, "doc:1"
    )
    result = finite_attribute_metrics(
        gold_attribute("race", "asian"), predictions, "race"
    )

    assert (result.true_positive, result.false_positive, result.false_negative) == (
        1,
        0,
        0,
    )


def test_conflicting_extra_finite_value_is_preserved_as_false_positive():
    predictions = predicted_attribute("race", "asian") + predicted_attribute(
        "race", "white", "doc:1"
    )
    result = finite_attribute_metrics(
        gold_attribute("race", "asian"), predictions, "race"
    )

    assert (result.true_positive, result.false_positive, result.false_negative) == (
        1,
        1,
        0,
    )


def test_missing_finite_value_is_false_negative():
    result = finite_attribute_metrics(
        gold_attribute("height", "5 feet 9 inches"),
        predicted_attribute("height", None),
        "height",
    )

    assert (result.true_positive, result.false_positive, result.false_negative) == (
        0,
        0,
        1,
    )


def test_finite_prediction_when_gold_is_null_is_false_positive():
    result = finite_attribute_metrics(
        gold_attribute("height", None),
        predicted_attribute("height", "5 feet 9 inches"),
        "height",
    )

    assert (result.true_positive, result.false_positive, result.false_negative) == (
        0,
        1,
        0,
    )


def test_multi_person_finite_values_are_pooled():
    gold_people = gold_attribute("gender", "male") + gold_attribute(
        "gender", "female", "doc:1"
    )
    predictions = predicted_attribute("gender", "male") + predicted_attribute(
        "gender", "female", "doc:1"
    )
    result = finite_attribute_metrics(gold_people, predictions, "gender")

    assert (result.true_positive, result.false_positive, result.false_negative) == (
        2,
        0,
        0,
    )


def test_multi_person_duplicate_finite_values_preserve_multiplicity():
    gold_people = gold_attribute("gender", "male") + gold_attribute(
        "gender", "male", "doc:1"
    )
    result = finite_attribute_metrics(
        gold_people, predicted_attribute("gender", "male"), "gender"
    )

    assert (result.true_positive, result.false_positive, result.false_negative) == (
        1,
        0,
        1,
    )


def test_height_does_not_use_approximate_matching():
    result = finite_attribute_metrics(
        gold_attribute("height", "5'11\""),
        predicted_attribute("height", "6 feet"),
        "height",
    )

    assert (result.true_positive, result.false_positive, result.false_negative) == (
        0,
        1,
        1,
    )


def test_height_curly_quotes_match_straight_quotes():
    result = finite_attribute_metrics(
        gold_attribute("height", "5'11\""),
        predicted_attribute("height", "5’11”"),
        "height",
    )

    assert (result.true_positive, result.false_positive, result.false_negative) == (
        1,
        0,
        0,
    )


def test_numeric_height_terminal_tall_and_whitespace_are_canonicalized():
    result = finite_attribute_metrics(
        gold_attribute("height", "6 feet"),
        predicted_attribute("height", "  6  feet tall  "),
        "height",
    )

    assert normalize_height_surface("  6  feet tall  ") == "6 feet"
    assert (result.true_positive, result.false_positive, result.false_negative) == (
        1,
        0,
        0,
    )


def test_height_does_not_use_numeric_tolerance():
    result = finite_attribute_metrics(
        gold_attribute("height", "5'10\""),
        predicted_attribute("height", "5'11\""),
        "height",
    )

    assert (result.true_positive, result.false_positive, result.false_negative) == (
        0,
        1,
        1,
    )


def test_corpus_finite_metrics_use_constant_gold_denominator():
    gold_index = {"doc": gold_attribute("gender", "male")}
    prediction_index = {
        "doc": predicted_attribute("gender", "male")
        + predicted_attribute("gender", "female", "doc:1")
    }
    result = evaluate_corpus(gold_index, prediction_index)

    assert result["gender"].true_positive + result["gender"].false_negative == 1
    assert corpus_gold_finite_values(gold_index, "gender") == 1


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
        metrics = evaluate_document(
            gold([{"item": gold_name, "descriptors": []}]),
            prediction([{"item": predicted_name, "descriptors": []}]),
        )
        for view in ("attr_only", "attr_value"):
            assert metrics[view].true_positive == int(equal)
            assert metrics[view].false_positive == int(not equal)
            assert metrics[view].false_negative == int(not equal)


def shared_gold(items_by_person, support="shared_identical", field="wearing.sweatshirt"):
    return [
        PersonAnnotation("doc", f"doc:{index}", None, {"wearing": items}, {
            "shared_group_tags": [{"group_id": "group", "support_type": support, "fields": [field]}],
        })
        for index, items in enumerate(items_by_person)
    ]


@pytest.mark.parametrize("support", ["shared_identical", "shared_subgroup"])
@pytest.mark.parametrize("prediction_count", [1, 5])
@pytest.mark.parametrize("field", ["wearing.sweatshirt", "wearing"])
def test_shared_clothing_is_one_primary_fact(support, prediction_count, field):
    gold_people = shared_gold([[item("sweatshirt", ["hooded"])]] * 5, support, field)
    predictions = [
        PersonPrediction("doc", f"doc:{index}", None, {"wearing": [item("sweatshirt", ["hooded"])]})
        for index in range(prediction_count)
    ]
    result = evaluate_document(gold_people, predictions)
    assert (result["attr_only"].true_positive, result["attr_only"].false_positive,
            result["attr_only"].false_negative) == (1, 0, 0)
    assert (result["attr_value"].true_positive, result["attr_value"].false_positive,
            result["attr_value"].false_negative) == (2, 0, 0)
    from hart.evaluation.extraction_metrics import gold_attr_value_units
    assert gold_attr_value_units(gold_people) == 2


def test_shared_primary_descriptors_keep_unsupported_evidence():
    result = evaluate_document(
        shared_gold([[item("sweatshirt", ["hooded"])]] * 5),
        prediction([item("sweatshirt", ["hooded", "hooded"])], "doc:0")
        + prediction([item("sweatshirt", ["black"])], "doc:1"),
    )
    assert result["attr_only"].f1 == 1.0
    assert (result["attr_value"].true_positive, result["attr_value"].false_positive,
            result["attr_value"].false_negative) == (2, 1, 0)


def test_distributed_primary_clothing_preserves_two_occurrences():
    result = evaluate_document(
        shared_gold([[item("hoodie", ["grey"])], [item("hoodie", ["dark"])]],
                    "group_distributed", "wearing.hoodie"),
        prediction([item("hoodie", ["dark"])], "doc:0")
        + prediction([item("hoodie", ["grey"])], "doc:1"),
    )
    assert result["attr_only"].true_positive == 2
    assert result["attr_value"].true_positive == 4
    assert all(metric.f1 == 1.0 for metric in result.values())


def test_unspecified_primary_descriptor_has_no_ownership_penalty():
    result = evaluate_document(
        shared_gold([[item("hoodie", ["bone texture"])], []],
                    "one_of_group_unspecified", "wearing.hoodie.descriptor:bone texture"),
        prediction([], "doc:0") + prediction([item("hoodie", ["bone texture"])], "doc:1"),
    )
    assert result["attr_value"].true_positive == 2
    assert all(metric.f1 == 1.0 for metric in result.values())


def test_untagged_primary_duplicates_still_count_as_separate_facts():
    result = evaluate_document(
        [PersonAnnotation("doc", f"doc:{index}", None, {"wearing": [item("sweatshirt", ["hooded"])]})
         for index in range(5)],
        prediction([item("sweatshirt", ["hooded"])]),
    )
    assert result["attr_only"].false_negative == 4
    assert result["attr_value"].false_negative == 8


def test_shared_primary_field_does_not_collapse_unrelated_same_name_item():
    gold_people = shared_gold([[item("sweatshirt", ["hooded"])]] * 5)
    gold_people += gold([item("sweatshirt", ["hooded"])], "doc:5")
    result = evaluate_document(
        gold_people,
        prediction([item("sweatshirt", ["hooded"])], "doc:0")
        + prediction([item("sweatshirt", ["hooded"])], "doc:5"),
    )
    assert result["attr_only"].true_positive == 2
    assert result["attr_value"].true_positive == 4
    assert all(metric.f1 == 1.0 for metric in result.values())
