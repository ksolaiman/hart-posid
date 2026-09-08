from hart.candidates import CandidateSentence, RegexSelector, split_sentences
from hart.extractors import POSIDExtractor
from hart.schema import PersonPrediction


SAMPLE = (
    "The suspect was described as a short Asian male, about 145 lbs. "
    "wearing a green jacket and blue pants."
)


class CandidateSelector:
    def __init__(self, sentence_ids, candidate_text=None, context_text=None):
        self.sentence_ids = sentence_ids
        self.candidate_text = candidate_text
        self.context_text = context_text
        self.calls = []

    def select(self, text, queries):
        self.calls.append((text, list(queries)))
        sentences = split_sentences(text)
        return [
            CandidateSentence(
                sentence_id=sentence_id,
                text=self.candidate_text or sentences[sentence_id],
                score=1.0,
                query=queries[0],
                context_text=self.context_text,
            )
            for sentence_id in self.sentence_ids
        ]


class Synset:
    def __init__(self, name):
        self.name = name

    def wup_similarity(self, other):
        return 0.8 if self.name == "color" and other.name.startswith("color:") else 0.2


class WordNet:
    colors = {"blue", "green", "grey", "purple", "red"}

    def synsets(self, token, *, pos):
        if token == "color":
            return [Synset("color")]
        if token in self.colors:
            return [Synset(f"color:{token}")]
        return [Synset(f"other:{token}")]


def tagger(tokens):
    colors = WordNet.colors
    return [
        (token, "JJ" if token.lower() in colors else "NN")
        for token in tokens
    ]


def extractor(selector, queries=("wearing",), threshold=0.7):
    return POSIDExtractor(
        selector,
        queries,
        wordnet=WordNet(),
        color_threshold=threshold,
        tagger=tagger,
    )


def test_posid_extracts_sample_properties_after_candidate_selection():
    selector = CandidateSelector(
        [1],
        candidate_text="wearing a green jacket and blue pants",
        context_text=SAMPLE,
    )
    result = extractor(selector).extract(SAMPLE, "doc")["doc"]

    assert selector.calls == [(SAMPLE, ["wearing"])]
    assert len(result) == 1
    assert isinstance(result[0], PersonPrediction)
    assert result[0].person_id == "doc:0"
    assert result[0].role == "suspect"
    assert result[0].attributes["gender"] == "male"
    assert result[0].attributes["race"] == "asian"
    assert result[0].attributes["height"] == "short"
    assert result[0].attributes["wearing"] == [
        {"item": "jacket", "descriptors": ["green"]},
        {"item": "pants", "descriptors": ["blue"]},
    ]
    assert "weight" not in result[0].attributes


def test_posid_reconstructed_symmetric_rules_and_legacy_height_forms():
    text = "The victim was a short Hispanic female wearing a purple coat."
    result = extractor(CandidateSelector([0])).extract(text, "doc")["doc"][0]

    assert result.role == "victim"
    assert result.attributes["gender"] == "female"
    assert result.attributes["race"] == "hispanic"
    assert result.attributes["height"] == "short"

    heights = extractor(CandidateSelector([0])).extract(
        "The suspect was 6 feet 2 inches tall wearing a red hat.", "height-doc"
    )["height-doc"][0].attributes["height"]
    assert heights == "6 feet 2 inches tall"


def test_posid_discovers_open_vocabulary_clothing_nouns():
    text = "The suspect was wearing a blue hoodie and red boots."
    people = extractor(CandidateSelector([0])).extract(text, "open")['open']

    assert people[0].attributes["wearing"] == [
        {"item": "hoodie", "descriptors": ["blue"]},
        {"item": "boots", "descriptors": ["red"]},
    ]


def test_posid_constructs_multiword_clothing_names_from_noun_runs():
    text = "The suspect was wearing a black tank top and grey dress pants."
    people = extractor(CandidateSelector([0])).extract(text, "multiword")['multiword']

    assert people[0].attributes["wearing"] == [
        {"item": "tank top", "descriptors": ["black"]},
        {"item": "dress pants", "descriptors": ["grey"]},
    ]


def test_posid_uses_candidate_scoped_fallback_without_wear_span():
    text = (
        "John is a 6-foot male wearing a red coat. "
        "The suspect was a short Asian male, blue hoodie and red boots."
    )
    people = extractor(CandidateSelector([1])).extract(text, "fallback")['fallback']

    assert len(people) == 1
    assert people[0].role == "suspect"
    assert people[0].attributes["gender"] == "male"
    assert people[0].attributes["race"] == "asian"
    assert people[0].attributes["height"] == "short"
    assert people[0].attributes["wearing"] == [
        {"item": "hoodie", "descriptors": ["blue"]},
        {"item": "boots", "descriptors": ["red"]},
    ]
    assert all(
        item["item"] != "coat"
        for item in people[0].attributes["wearing"]
    )


def test_posid_only_extracts_clothing_from_selected_candidates():
    text = "A red coat was discarded. The suspect was wearing blue pants."
    result = extractor(CandidateSelector([1])).extract(text, "doc")["doc"]

    wearing = [item for person in result for item in person.attributes.get("wearing", [])]
    assert wearing == [{"item": "pants", "descriptors": ["blue"]}]


def test_posid_does_not_leak_fixed_properties_from_noncandidate_sentence():
    text = "John is a 6-foot male. The suspect was wearing blue pants."
    selector = RegexSelector()

    candidates = selector.select(text, ["wearing"])
    assert len(candidates) == 1
    assert candidates[0].text == "wearing blue pants"
    assert candidates[0].context_text == "The suspect was wearing blue pants."

    people = extractor(selector).extract(text, "noncandidate")["noncandidate"]

    assert len(people) == 1
    assert people[0].person_id == "noncandidate:0"
    assert people[0].role == "suspect"
    assert people[0].attributes["wearing"] == [
        {"item": "pants", "descriptors": ["blue"]}
    ]
    assert "height" not in people[0].attributes
    assert "gender" not in people[0].attributes


def test_posid_keeps_same_sentence_john_and_mike_attributes_distinct():
    text = "John is a 6-foot male and Mike is a 5'8\" male."
    people = extractor(CandidateSelector([0])).extract(text, "two-people")[
        "two-people"
    ]

    assert len(people) == 2
    assert [person.person_id for person in people] == [
        "two-people:0",
        "two-people:1",
    ]
    assert people[0].attributes["gender"] == "male"
    assert people[0].attributes["height"] == "6-foot"
    assert people[1].attributes["gender"] == "male"
    assert people[1].attributes["height"] == '5\'8"'


def test_posid_wordnet_color_threshold_is_strict():
    result = extractor(CandidateSelector([0]), threshold=0.8).extract(
        "The suspect was wearing a green jacket.", "doc"
    )["doc"][0]

    assert result.attributes["wearing"][0]["descriptors"] == []


def test_posid_passes_coordination_cues_to_person_resolution():
    coordinated = extractor(CandidateSelector([1])).extract(
        "John and Mike entered the store. Both were wearing black jackets.",
        "shared",
    )["shared"]
    assert len(coordinated) == 2
    assert all(
        person.attributes.get("wearing")
        == [{"item": "jacket", "descriptors": ["black"]}]
        for person in coordinated
    )
    assert all(person.attributes.get("race") != "black" for person in coordinated)


def test_posid_passes_pronoun_cue_to_person_resolution():
    pronoun = extractor(CandidateSelector([2])).extract(
        "John is a male. The suspect is a male. He wore a blue coat.", "pronoun"
    )["pronoun"]
    assert "wearing" not in pronoun[0].attributes
    assert pronoun[1].attributes["wearing"][0]["item"] == "coat"


def test_posid_does_not_extract_when_candidate_selector_misses():
    selector = CandidateSelector([])
    assert extractor(selector).extract("A male.", "doc") == {"doc": []}
