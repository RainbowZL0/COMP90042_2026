"""Retriever interface contract.

Phase 3 will add a reranker-augmented retriever; defining the abstract
interface here means the Pipeline does not need to change when we swap
in the new component.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalResult:
    """A single (evidence_id, score) pair from a retriever."""

    evidence_id: str
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_id, str) or not self.evidence_id:
            raise ValueError("evidence_id must be a non-empty str")


class Retriever(ABC):
    """Contract for any retrieval component (BM25, reranker, hybrid, ...).

    All retrievers MUST implement `retrieve_batch`. The single-claim
    `retrieve` is provided as a thin wrapper for convenience.
    """

    @abstractmethod
    def retrieve_batch(
        self, claim_texts: list[str], top_k: int
    ) -> list[list[RetrievalResult]]:
        """Return top_k evidence per claim, ranked by descending score."""

    def retrieve(self, claim_text: str, top_k: int) -> list[RetrievalResult]:
        return self.retrieve_batch([claim_text], top_k)[0]
