from __future__ import annotations

from hart.candidates import RegexSelector, Word2VecSelector, WordNetSelector, ZeroShotSelector


class FakeVectors:
    values = {
        "red": [1.0, 0.0],
        "dress": [1.0, 0.0],
        "blue": [0.0, 1.0],
        "car": [0.0, 1.0],
    }

    def __getitem__(self, token):
        return self.values[token]


class FakeSynset:
    def __init__(self, name):
        self.name = name

    def path_similarity(self, other):
        return 1.0 if self.name == other.name else 0.0


class FakeWordNet:
    def synsets(self, token, **kwargs):
        return [FakeSynset(token)]


def test_regex_selector_matches_once_per_sentence():
    results = RegexSelector().select("A red dress. A blue car.", ["dress", "car"])
    assert [result.sentence_id for result in results] == [0, 1]


def test_word2vec_selector_uses_configured_threshold():
    results = Word2VecSelector(FakeVectors(), threshold=0.9).select("A red dress. A blue car.", ["red"])
    assert [result.sentence_id for result in results] == [0]


def test_wordnet_selector_uses_injected_wordnet():
    results = WordNetSelector(threshold=1.0, wordnet=FakeWordNet()).select(
        "A red dress. A dress.", iter(["dress"])
    )
    assert [result.sentence_id for result in results] == [0, 1]
    assert all(result.query == "dress" for result in results)


def test_zero_shot_selector_uses_injected_classifier():
    def classifier(text, candidate_labels, multi_label):
        return {"labels": candidate_labels, "scores": [0.8, 0.2]}

    results = ZeroShotSelector(classifier, threshold=0.75).select("A red dress.", ["clothing", "vehicle"])
    assert len(results) == 1
    assert results[0].score == 0.8
