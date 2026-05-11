"""Classifier interface contract.

Phase 4 will replace MajorityClassifier with the real verifier
(per-evidence stance + attention pooling). The Pipeline talks only to
this ABC so the swap is transparent.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from src.data.schema import ClaimLabel, Evidence


class Classifier(ABC):
    """Contract for any claim classifier."""

    @abstractmethod
    def classify_batch(
        self,
        claim_texts: Sequence[str],
        evidences_batch: Sequence[Sequence[Evidence]],
    ) -> list[ClaimLabel]:
        """Predict a label for each (claim, evidences) pair."""

    def classify(
        self, claim_text: str, evidences: Sequence[Evidence]
    ) -> ClaimLabel:
        return self.classify_batch([claim_text], [evidences])[0]
