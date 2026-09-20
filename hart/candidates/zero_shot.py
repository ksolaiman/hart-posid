from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .base import CandidateSelector, CandidateSentence, split_sentences


class ZeroShotSelector(CandidateSelector):
    """Select sentences classified by an injected zero-shot pipeline."""

    def __init__(self, classifier: Any, threshold: float | None = None, multi_label: bool = False):
        self.classifier = classifier
        self.threshold = threshold
        self.multi_label = multi_label

    def select(self, text: str, queries: Iterable[str]) -> list[CandidateSentence]:
        labels = list(queries)
        if not labels:
            return []
        for sentence_id, sentence in enumerate(split_sentences(text)):
            output = self.classifier(
                sentence,
                candidate_labels=labels,
                # hypothesis_template="This text is about {}",        # legacy - 2019 - "This example is {}", (hugging face default)
                multi_label=self.multi_label,
            )
            scores = dict(zip(output["labels"], output["scores"]))
            query, score = max(scores.items(), key=lambda item: item[1])
             
            if self.threshold is None or score > self.threshold:
                return [CandidateSentence(sentence_id, sentence, score, query, context_text=sentence)]
        return []
