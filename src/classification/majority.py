"""Majority-class classifier.

Phase 2 placeholder: always predicts the most frequent label seen in
training. Its purpose is purely to complete the end-to-end pipeline so
we can wire up retrieval + I/O + eval.py. The real classifier is built
in Phase 4.

This also serves as the baseline-classifier component in our final
report's ablation: comparing (BM25 + Majority) vs (BM25 + Real Classifier)
isolates the contribution of the classifier alone.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

from src.classification.base import Classifier
from src.data.schema import Claim, ClaimLabel, Evidence


class MajorityClassifier(Classifier):
    """Predicts a fixed label for every input."""

    def __init__(self, label: ClaimLabel = ClaimLabel.SUPPORTS):
        if not isinstance(label, ClaimLabel):
            raise TypeError(f"label must be ClaimLabel, got {type(label).__name__}")
        self._label = label

    @classmethod
    def fit(cls, train_claims: dict[str, Claim]) -> "MajorityClassifier":
        """Fit by counting label frequencies on labeled training data."""
        labels = [c.label for c in train_claims.values() if c.is_labeled]
        if not labels:
            raise ValueError("Cannot fit majority classifier on unlabeled data")
        most_common_label, _ = Counter(labels).most_common(1)[0]
        return cls(label=most_common_label)

    @property
    def label(self) -> ClaimLabel:
        return self._label

    def classify_batch(
        self,
        claim_texts: Sequence[str],
        evidences_batch: Sequence[Sequence[Evidence]],
    ) -> list[ClaimLabel]:
        if len(claim_texts) != len(evidences_batch):
            raise ValueError(
                f"claim_texts and evidences_batch must have same length "
                f"({len(claim_texts)} vs {len(evidences_batch)})"
            )
        return [self._label] * len(claim_texts)
