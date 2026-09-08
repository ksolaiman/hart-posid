from hart.candidates import CandidateSentence, StackedSelector, WordNetSelector


class FakeSelector:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def select(self, text, queries):
        self.calls.append((text, list(queries)))
        return self.results


def test_stacked_returns_primary_results_without_calling_fallback():
    primary_result = [CandidateSentence(0, "primary", 1.0, "wearing")]
    primary = FakeSelector(primary_result)
    fallback = FakeSelector([CandidateSentence(1, "fallback", 1.0, "clothes")])
    selector = StackedSelector(primary, fallback, ["wearing"], ["clothes"])

    result = selector.select("text", ["ignored"])
    assert result == primary_result
    assert result[0] is primary_result[0]
    assert primary.calls == [("text", ["wearing"])]
    assert fallback.calls == []


def test_stacked_uses_fallback_queries_after_primary_miss():
    fallback_result = [CandidateSentence(1, "fallback", 0.9, "clothes")]
    primary = FakeSelector([])
    fallback = FakeSelector(fallback_result)
    selector = StackedSelector(primary, fallback, ["wearing"], ["clothes"])

    assert selector.select("text", ["ignored"]) == fallback_result
    assert primary.calls == [("text", ["wearing"])]
    assert fallback.calls == [("text", ["clothes"])]


class Synset:
    def __init__(self, scores):
        self.scores = scores

    def wup_similarity(self, other):
        return self.scores.get(other)


class LegacyWordNet:
    def __init__(self):
        self.low = object()
        self.qualifying = object()
        self.higher = object()
        self.first_query = Synset(
            {self.low: 0.4, self.qualifying: 0.91, self.higher: 0.99}
        )
        self.unused_query = Synset({self.low: 1.0})

    def synsets(self, token, *, pos):
        return {
            "clothes": [self.first_query, self.unused_query],
            "jacket": [self.low, self.qualifying],
            "pants": [self.higher],
        }.get(token.lower(), [])


def test_verified_wordnet_mode_uses_first_query_all_token_synsets_and_first_hit():
    result = WordNetSelector(
        threshold=0.9,
        wordnet=LegacyWordNet(),
        synset_strategy="legacy_asymmetric",
    ).select("A jacket and pants.", ["clothes"])

    assert len(result) == 1
    assert result[0].score == 0.91
    assert result[0].text == "A jacket and pants."


def test_verified_wordnet_mode_handles_query_without_noun_synset():
    result = WordNetSelector(
        threshold=0.9,
        wordnet=LegacyWordNet(),
        synset_strategy="legacy_asymmetric",
    ).select("A jacket.", ["unknown-query"])

    assert result == []
