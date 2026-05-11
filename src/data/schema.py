"""Core data structures for claims, evidences, and predictions.

These dataclasses are the canonical representation passed between modules.
Validation lives in __post_init__ to catch malformed data at construction
time rather than at use time.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ClaimLabel(str, Enum):
    """The four valid claim labels per the assignment spec."""

    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"
    DISPUTED = "DISPUTED"

    @classmethod
    def values(cls) -> list[str]:
        return [m.value for m in cls]


@dataclass(frozen=True)
class Evidence:
    """A single evidence passage from the corpus."""

    id: str
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError(f"Evidence id must be a non-empty str, got {self.id!r}")
        if not isinstance(self.text, str):
            raise TypeError(
                f"Evidence text must be str, got {type(self.text).__name__}"
            )


@dataclass(frozen=True)
class Claim:
    """A claim. May or may not be labeled (test set is unlabeled)."""

    id: str
    text: str
    label: Optional[ClaimLabel] = None
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError(f"Claim id must be a non-empty str, got {self.id!r}")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError(f"Claim text must be a non-empty str, got {self.text!r}")
        if self.label is not None and not isinstance(self.label, ClaimLabel):
            raise TypeError(
                f"Claim label must be ClaimLabel or None, "
                f"got {type(self.label).__name__}"
            )
        if not isinstance(self.evidence_ids, tuple):
            raise TypeError(
                f"Claim evidence_ids must be tuple, "
                f"got {type(self.evidence_ids).__name__}"
            )

    @property
    def is_labeled(self) -> bool:
        return self.label is not None


@dataclass(frozen=True)
class Prediction:
    """System output for a single claim. Always labeled, always has >=1 evidence."""

    claim_id: str
    claim_text: str
    claim_label: ClaimLabel
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.claim_id, str) or not self.claim_id:
            raise ValueError(f"Prediction claim_id must be non-empty str")
        if not isinstance(self.claim_label, ClaimLabel):
            raise TypeError(
                f"Prediction claim_label must be ClaimLabel, "
                f"got {type(self.claim_label).__name__}"
            )
        if not isinstance(self.evidence_ids, tuple) or len(self.evidence_ids) == 0:
            # eval.py requires at least one evidence per claim (assignment spec).
            raise ValueError("Prediction must contain at least one evidence id")
