from hart.extractors import DependencyExtractor, RegexBaseline
from hart.schema import PersonPrediction


def test_regex_baseline_emits_person_level_predictions():
    result = RegexBaseline().extract("John is a 6-foot male. The suspect is a 5'8\" male.", "doc")["doc"]
    assert len(result) == 2
    assert all(isinstance(person, PersonPrediction) for person in result)
    assert [person.person_id for person in result] == ["doc:0", "doc:1"]
    assert result[0].attributes["height"] == "6-foot"
    assert result[0].attributes["gender"] == "male"
    assert result[1].role == "suspect"
    assert result[1].attributes["height"] == '5\'8"'


def test_regex_baseline_attaches_pronoun_to_most_recent_person():
    result = RegexBaseline().extract(
        "John is a 6-foot male. The suspect is a 5'8\" male. "
        "He was wearing a red jacket.",
        "doc",
    )["doc"]

    assert len(result) == 2
    assert "wearing" not in result[0].attributes
    assert result[1].attributes["wearing"] == [
        {"item": "jacket", "descriptors": ["red"]}
    ]


def test_regex_baseline_handles_coordination_and_anonymous_people():
    coordinated = RegexBaseline().extract(
        "John and Mike entered the store. Both were wearing black jackets.",
        "shared",
    )["shared"]
    assert len(coordinated) == 2
    assert all(
        person.attributes["wearing"]
        == [{"item": "jacket", "descriptors": ["black"]}]
        for person in coordinated
    )

    anonymous = RegexBaseline().extract(
        "A 5'9\" male wearing blue pants was seen nearby.", "anonymous"
    )["anonymous"]
    assert len(anonymous) == 1
    assert anonymous[0].person_id == "anonymous:0"
    assert anonymous[0].attributes["height"] == '5\'9"'
    assert anonymous[0].attributes["wearing"][0]["item"] == "pants"


def test_regex_baseline_does_not_extract_race():
    result = RegexBaseline().extract(
        "The suspect was an Asian male wearing a blue coat.", "doc"
    )["doc"][0]
    assert "race" not in result.attributes


class Token:
    def __init__(
        self,
        text,
        idx,
        children=(),
        like_num=False,
        pos_="",
        dep_="",
        head=None,
    ):
        self.text = text
        self.idx = idx
        self.children = children
        self.like_num = like_num
        self.pos_ = pos_
        self.dep_ = dep_
        self.head = head


class Sentence(list):
    def __init__(self, tokens, text):
        super().__init__(tokens)
        self._text = text

    @property
    def text(self):
        return self._text


class Doc:
    def __init__(self):
        adjective = Token("red", 18, pos_="ADJ")
        self.sents = [
            Sentence(
                [
                    Token("The", 0),
                    Token("suspect", 4),
                    Token("wears", 12),
                    adjective,
                    Token("jacket", 22, children=[adjective]),
                    Token("male", 34),
                ],
                "The suspect wears a red jacket and is male.",
            )
        ]


class HeightDoc:
    def __init__(self):
        tall = Token("tall", 31, pos_="ADJ", dep_="acomp")
        feet = Token("feet", 17, pos_="NOUN", dep_="npadvmod", head=tall)
        inches = Token("inches", 24, pos_="NOUN", dep_="npadvmod", head=tall)
        six = Token("6", 15, like_num=True, pos_="NUM", dep_="nummod", head=feet)
        two = Token("2", 22, like_num=True, pos_="NUM", dep_="nummod", head=inches)
        feet.children = [six]
        inches.children = [two]
        tall.children = [feet, inches]
        self.sents = [
            Sentence(
                [
                    Token("The", 0),
                    Token("suspect", 4),
                    Token("is", 12),
                    six,
                    feet,
                    two,
                    inches,
                    tall,
                    Token(".", 35),
                ],
                "The suspect is 6 feet 2 inches tall.",
            )
        ]


class NoNumericHeightDoc:
    def __init__(self):
        self.sents = [
            Sentence(
                [
                    Token("The", 0),
                    Token("suspect", 4),
                    Token("is", 12),
                    Token("tall", 15, pos_="ADJ", dep_="acomp"),
                    Token(".", 19),
                ],
                "The suspect is tall.",
            )
        ]


def test_dependency_baseline_uses_injected_parsed_document():
    result = DependencyExtractor(lambda text: Doc()).extract("ignored", "doc")["doc"]
    assert len(result) == 1
    assert isinstance(result[0], PersonPrediction)
    assert result[0].role == "suspect"
    assert result[0].attributes["gender"] == "male"
    assert result[0].attributes["wearing"][0]["item"] == "jacket"


def test_dependency_baseline_extracts_numeric_unit_height_structure():
    result = DependencyExtractor(lambda text: HeightDoc()).extract(
        "ignored", "height-doc"
    )["height-doc"]

    assert len(result) == 1
    assert result[0].role == "suspect"
    assert "6 feet 2 inches" in result[0].attributes["height"]

    without_numeric_units = DependencyExtractor(
        lambda text: NoNumericHeightDoc()
    ).extract("ignored", "control")["control"]
    assert len(without_numeric_units) == 1
    assert "height" not in without_numeric_units[0].attributes
