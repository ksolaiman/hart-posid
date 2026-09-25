from hart.candidates import CandidateSentence, StackedSelector, WordNetSelector


class FakeSelector:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def select(self, text, queries):
        self.calls.append((text, list(queries)))
        return self.results


def test_stacked_combines_primary_and_semantic_results():
    primary_result = [CandidateSentence(0, "primary", 1.0, "wearing")]
    primary = FakeSelector(primary_result)
    fallback = FakeSelector([CandidateSentence(1, "fallback", 1.0, "clothes")])
    selector = StackedSelector(primary, fallback, ["wearing"], ["clothes"])

    result = selector.select("text", ["ignored"])
    assert result == primary_result + fallback.results
    assert result[0] is primary_result[0]
    assert primary.calls == [("text", ["wearing"])]
    assert fallback.calls == [("text", ["clothes"])]


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


def test_stacked_preserves_roles_and_posid_deduplicates_equivalent_work(monkeypatch):
    from hart.extractors import POSIDExtractor

    text = "The suspect was wearing a red jacket."
    wearing = CandidateSentence(0, text, 1.0, "wearing", context_text=text)
    fixed = CandidateSentence(0, text, 1.0, "regex:suspect", context_text=text)
    semantic = CandidateSentence(0, text, 0.95, "clothes", context_text=text)
    primary = FakeSelector([fixed, wearing, wearing])
    fallback = FakeSelector([semantic, semantic])
    selector = StackedSelector(primary, fallback, ["wearing"], ["clothes"])
    extractor = POSIDExtractor(selector, [])
    calls = []

    def extract_sentence(sentence, sentence_id, **kwargs):
        calls.append((sentence_id, sentence, kwargs["include_fixed"], kwargs["include_clothing"]))
        return []

    monkeypatch.setattr(extractor, "_extract_sentence", extract_sentence)
    assert extractor.extract(text, "doc") == {"doc": []}
    assert calls == [
        (0, text, True, False),
        (0, text, True, True),
        (0, text, False, True),
    ]
    assert fallback.calls == [(text, ["clothes"])]


def test_stacked_keeps_regex_candidates_with_first_match_semantic_mode():
    text = "A jacket. Some pants."
    primary_results = [CandidateSentence(0, "A jacket.", 1.0, "wearing")]
    semantic = WordNetSelector(
        threshold=0.9, wordnet=LegacyWordNet(), synset_strategy="legacy_asymmetric",
        candidate_selector_non_re_first_match_only=True,
    )
    selector = StackedSelector(FakeSelector(primary_results), semantic, ["wearing"], ["clothes"])
    results = selector.select(text)
    assert results == primary_results + semantic.select(text, ["clothes"])
    assert len(results) == 2
    assert [candidate.query for candidate in results] == ["wearing", "clothes"]
