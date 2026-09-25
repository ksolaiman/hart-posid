from __future__ import annotations

import pytest

from hart.candidates import RegexSelector, Word2VecSelector, WordNetSelector, ZeroShotSelector


class FakeVectors:
    def similarity(self, token, query):
        if token.lower() not in {"red", "dress", "blue", "car"}:
            raise KeyError(token)
        return 1.0 if token.lower() == query else 0.0


class FakeSynset:
    def __init__(self, name):
        self.name = name

    def wup_similarity(self, other):
        return 1.0 if self.name == other.name else 0.0


class FakeWordNet:
    def synsets(self, token, **kwargs):
        return [FakeSynset(token)] if token.isalpha() else []


def test_regex_selector_matches_once_per_sentence():
    results = RegexSelector().select("A red dress. A blue car.", ["dress", "car"])
    assert [result.sentence_id for result in results] == [0, 1]


@pytest.mark.parametrize("first_only", [None, False, True])
def test_word2vec_selector_uses_configured_threshold(first_only):
    options = {} if first_only is None else {"candidate_selector_non_re_first_match_only": first_only}
    results = Word2VecSelector(FakeVectors(), threshold=1.0, **options).select("A red dress. A blue car. Another red dress.", ["red"])
    assert [result.sentence_id for result in results] == ([0] if first_only else [0, 2])
    assert [result.score for result in results] == ([1.0] if first_only else [1.0, 1.0])
    assert all(result.context_text == result.text and result.query == "red" for result in results)


@pytest.mark.parametrize("strategy", ["first", "max_all", "average", "legacy_asymmetric"])
@pytest.mark.parametrize("first_only", [None, False, True])
def test_wordnet_selector_uses_injected_wordnet(strategy, first_only):
    options = {} if first_only is None else {"candidate_selector_non_re_first_match_only": first_only}
    results = WordNetSelector(threshold=1.0, wordnet=FakeWordNet(), synset_strategy=strategy, **options).select(
        "Dress. Dress.", iter(["dress"])
    )
    assert [result.sentence_id for result in results] == ([0] if first_only else [0, 1])
    assert all(result.query == "dress" for result in results)


@pytest.mark.parametrize("first_only", [None, False, True])
def test_zero_shot_selector_uses_injected_classifier(first_only):
    options = {} if first_only is None else {"candidate_selector_non_re_first_match_only": first_only}
    calls = []

    def classifier(text, candidate_labels, hypothesis_template, multi_label):
        calls.append(
            (
                text,
                [hypothesis_template.format(query) for query in candidate_labels],
                multi_label,
            )
        )
        score = 0.75 if text == "First sentence." else 0.8
        return {"labels": candidate_labels, "scores": [score, 0.2]}

    results = ZeroShotSelector(classifier, threshold=0.75, **options).select(
        "First sentence. Second sentence. Third sentence.",
        ["clothing", "vehicle"],
    )

    assert [result.sentence_id for result in results] == ([1] if first_only else [1, 2])
    assert results[0].score == 0.8
    assert calls == [
        (
            "First sentence.",
            ["This example is clothing.", "This example is vehicle."],
            False,
        ),
        (
            "Second sentence.",
            ["This example is clothing.", "This example is vehicle."],
            False,
        ),
        (
            "Third sentence.",
            ["This example is clothing.", "This example is vehicle."],
            False,
        ),
    ][:2 if first_only else 3]
