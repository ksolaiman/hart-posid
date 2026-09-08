from hart.extractors.person_resolution import AttributeMention, resolve_mentions


def test_named_people_remain_distinct_and_ids_are_deterministic():
    result = resolve_mentions(
        "doc",
        [
            AttributeMention("height", "6-foot", 0, 5, "John"),
            AttributeMention("gender", "male", 0, 14, "John"),
            AttributeMention("height", "5'8\"", 1, 20, "suspect", "suspect"),
            AttributeMention("gender", "male", 1, 28, "suspect", "suspect"),
        ],
    )["doc"]

    assert [person.person_id for person in result] == ["doc:0", "doc:1"]
    assert result[0].attributes["height"] == "6-foot"
    assert result[1].role == "suspect"
    assert result[1].attributes["height"] == '5\'8"'


def test_pronoun_and_coordination_attach_locally():
    result = resolve_mentions(
        "doc",
        [
            AttributeMention("__anchor__", None, 0, 0, "John"),
            AttributeMention("__anchor__", None, 0, 9, "Mike"),
            AttributeMention("wearing", {"item": "jacket", "descriptors": ["black"]}, 1, 20, shared_anchors=("John", "Mike")),
        ],
    )["doc"]
    assert len(result) == 2
    assert all(person.attributes["wearing"] for person in result)

    pronoun_result = resolve_mentions(
        "doc",
        [
            AttributeMention("__anchor__", None, 0, 0, "suspect", "suspect"),
            AttributeMention("gender", "male", 0, 8, "suspect", "suspect"),
            AttributeMention("wearing", {"item": "jacket", "descriptors": ["red"]}, 1, 3, pronoun="he"),
        ],
    )["doc"]
    assert pronoun_result[0].attributes["wearing"][0]["item"] == "jacket"


def test_pronoun_uses_most_recent_compatible_person():
    result = resolve_mentions(
        "doc",
        [
            AttributeMention("gender", "male", 0, 5, "John"),
            AttributeMention("gender", "male", 1, 8, "suspect", "suspect"),
            AttributeMention(
                "wearing",
                {"item": "jacket", "descriptors": ["red"]},
                2,
                3,
                pronoun="he",
            ),
        ],
    )["doc"]

    assert "wearing" not in result[0].attributes
    assert result[1].attributes["wearing"][0]["descriptors"] == ["red"]


def test_unanchored_mention_uses_nearest_active_person_without_merging_people():
    result = resolve_mentions(
        "doc",
        [
            AttributeMention("height", "6-foot", 0, 5, "John"),
            AttributeMention("height", '5\'8"', 1, 8, "suspect", "suspect"),
            AttributeMention("gender", "male", 2, 4),
        ],
    )["doc"]

    assert len(result) == 2
    assert "gender" not in result[0].attributes
    assert result[1].attributes["gender"] == "male"


def test_attributes_without_an_anchor_create_anonymous_person():
    result = resolve_mentions(
        "doc",
        [AttributeMention("gender", "male"), AttributeMention("height", "5'9\"")],
    )["doc"]
    assert result[0].person_id == "doc:0"
    assert result[0].attributes["gender"] == "male"
