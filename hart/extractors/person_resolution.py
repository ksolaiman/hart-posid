from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable
import re
from typing import Any

from hart.schema import PersonPrediction


PERSON_ROLES = ("suspect", "victim", "juvenile")
_NON_NAME_WORDS = {
    "a", "an", "the", "both", "he", "she", "they", "male", "female",
    "man", "woman", "boy", "girl", "asian", "black", "white", "hispanic",
    "latino", "latina",
}


def find_person_anchors(sentence: str) -> list[tuple[str, int, str | None]]:
    """Find narrow explicit role and proper-name anchors in textual order."""
    anchors: list[tuple[str, int, str | None]] = []
    for role in PERSON_ROLES:
        for match in re.finditer(rf"\b{role}\b", sentence, re.IGNORECASE):
            anchors.append((match.group(0), match.start(), role))
    for match in re.finditer(r"\b[A-Z][a-z]+\b", sentence):
        if match.group(0).lower() not in _NON_NAME_WORDS | set(PERSON_ROLES):
            anchors.append((match.group(0), match.start(), None))
    return sorted(anchors, key=lambda item: item[1])


def nearest_preceding_anchor(
    anchors: Iterable[tuple[str, int, str | None]],
    position: int,
) -> tuple[str, int, str | None] | None:
    preceding = [anchor for anchor in anchors if anchor[1] <= position]
    return preceding[-1] if preceding else None


@dataclass(frozen=True)
class AttributeMention:
    attribute: str
    value: Any
    sentence_id: int = 0
    position: int = 0
    anchor: str | None = None
    role: str | None = None
    pronoun: str | bool | None = None
    shared_anchors: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None


@dataclass
class _PersonState:
    key: str
    role: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_sentence: int = -1
    last_position: int = -1


def _normal_key(value: str) -> str:
    return " ".join(value.lower().split())


def _add_attribute(attributes: dict[str, Any], name: str, value: Any) -> None:
    if name == "wearing":
        existing = attributes.setdefault("wearing", [])
        if isinstance(value, list):
            values = value
        else:
            values = [value]
        for item in values:
            if item not in existing:
                existing.append(item)
        return
    attributes[name] = value


class PersonResolver:
    """Resolve local attribute mentions into deterministic person records."""

    def resolve(
        self,
        document_id: str,
        mentions: Iterable[AttributeMention],
    ) -> dict[str, list[PersonPrediction]]:
        ordered = sorted(
            mentions,
            key=lambda mention: (mention.sentence_id, mention.position),
        )
        people: list[_PersonState] = []
        by_anchor: dict[str, _PersonState] = {}

        for mention in ordered:
            targets: list[_PersonState] = []
            if mention.shared_anchors:
                for anchor in mention.shared_anchors:
                    targets.append(self._person_for_anchor(anchor, people, by_anchor))
                resolution = "shared_coordination"
            elif mention.anchor:
                targets = [self._person_for_anchor(mention.anchor, people, by_anchor, mention.role)]
                resolution = "explicit"
            else:
                active = self._local_candidates(people, mention)
                if isinstance(mention.pronoun, str):
                    compatible = [
                        person
                        for person in active
                        if self._pronoun_compatible(person, mention.pronoun)
                    ]
                    if compatible:
                        active = compatible
                if mention.pronoun and active:
                    targets = [active[0]]
                    resolution = "pronoun"
                elif len(active) == 1:
                    targets = active
                    resolution = "same_sentence" if active[0].last_sentence == mention.sentence_id else "nearest_local"
                elif active:
                    targets = [active[0]]
                    resolution = "nearest_local"
                else:
                    person = _PersonState(f"anonymous:{len(people)}")
                    people.append(person)
                    targets = [person]
                    resolution = "anonymous"

            for person in targets:
                if mention.attribute != "__anchor__":
                    _add_attribute(person.attributes, mention.attribute, mention.value)
                person.last_sentence = mention.sentence_id
                person.last_position = mention.position
                person.metadata.setdefault("resolution", []).append(
                    mention.metadata.get("resolution", resolution)
                    if mention.metadata
                    else resolution
                )

        predictions = []
        for index, person in enumerate(people):
            person_id = f"{document_id}:{index}"
            metadata = dict(person.metadata) or None
            predictions.append(
                PersonPrediction(
                    document_id=document_id,
                    person_id=person_id,
                    role=person.role,
                    attributes=person.attributes,
                    metadata=metadata,
                )
            )
        return {document_id: predictions}

    @staticmethod
    def _person_for_anchor(
        anchor: str,
        people: list[_PersonState],
        by_anchor: dict[str, _PersonState],
        role: str | None = None,
    ) -> _PersonState:
        key = _normal_key(anchor)
        if key in by_anchor:
            person = by_anchor[key]
            if person.role is None and role is not None:
                person.role = role
            return person
        person = _PersonState(key=key, role=role)
        people.append(person)
        by_anchor[key] = person
        return person

    @staticmethod
    def _local_candidates(
        people: list[_PersonState],
        mention: AttributeMention,
    ) -> list[_PersonState]:
        active = [person for person in people if person.last_sentence >= 0]
        same_sentence = [
            person for person in active if person.last_sentence == mention.sentence_id
        ]
        candidates = same_sentence or active
        return sorted(
            candidates,
            key=lambda person: (person.last_sentence, person.last_position),
            reverse=True,
        )

    @staticmethod
    def _pronoun_compatible(person: _PersonState, pronoun: str) -> bool:
        gender = person.attributes.get("gender")
        if pronoun.lower() == "he":
            return gender in {None, "male"}
        if pronoun.lower() == "she":
            return gender in {None, "female"}
        return True


def resolve_mentions(
    document_id: str,
    mentions: Iterable[AttributeMention],
) -> dict[str, list[PersonPrediction]]:
    return PersonResolver().resolve(document_id, mentions)


resolve_person_mentions = resolve_mentions
