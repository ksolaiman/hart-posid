from __future__ import annotations

from hart.candidates import RegexSelector, Word2VecSelector, WordNetSelector, ZeroShotSelector


SAMPLE_TEXT = (
    "The suspect was described as a short Asian male, about 145 lbs. "
    "wearing a green jacket and blue pants."
)
EXPECTED_CLOTHING_SENTENCE = "wearing a green jacket and blue pants."


class SampleVectors:
    """Deterministic injected double; its values are not resource measurements."""

    def similarity(self, token, query):
        if token.lower() != "wearing" or query != "clothes":
            raise KeyError(token)
        return 0.5


class SampleSynset:
    def __init__(self, name):
        self.name = name

    def wup_similarity(self, other):
        if {self.name, other.name} == {"wearing", "clothes"}:
            return 0.95
        return None


class SampleWordNet:
    def synsets(self, token, *, pos):
        assert pos == "n"
        if token in {"wearing", "clothes"}:
            return [SampleSynset(token)]
        return []


def sample_classifier(text, candidate_labels, hypothesis_template, multi_label):
    assert candidate_labels == ["clothes"]
    assert hypothesis_template.format(candidate_labels[0]) == "This example is clothes."
    assert multi_label is False
    score = 0.86 if "wearing" in text else 0.85
    return {"labels": candidate_labels, "scores": [score]}


def test_candidate_selectors_with_legacy_experiment_settings():
    regex_results = RegexSelector().select(SAMPLE_TEXT, ["wearing"])
    word2vec_results = Word2VecSelector(SampleVectors(), threshold=0.5).select(
        SAMPLE_TEXT,
        ["clothes"],
    )
    wordnet_results = WordNetSelector(
        threshold=0.9,
        wordnet=SampleWordNet(),
        synset_strategy="legacy_asymmetric",
    ).select(SAMPLE_TEXT, ["clothes"])
    zero_shot_results = ZeroShotSelector(
        sample_classifier,
        threshold=0.85,
    ).select(SAMPLE_TEXT, ["clothes"])

    assert [candidate.text for candidate in regex_results] == [
        "wearing a green jacket and blue pants"
    ]
    assert [candidate.context_text for candidate in regex_results] == [SAMPLE_TEXT]
    assert [candidate.text for candidate in word2vec_results] == [EXPECTED_CLOTHING_SENTENCE]
    assert [candidate.text for candidate in wordnet_results] == [EXPECTED_CLOTHING_SENTENCE]
    assert [candidate.text for candidate in zero_shot_results] == [EXPECTED_CLOTHING_SENTENCE]
