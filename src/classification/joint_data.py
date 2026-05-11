"""Data helpers for the joint evidence classifier.

The joint classifier treats claim verification as a 4-way supervised problem:

    input:  claim text + a small bundle of evidence passages
    output: SUPPORTS / REFUTES / NOT_ENOUGH_INFO / DISPUTED

This module deliberately stays model-free so it can be tested without torch or
transformers installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from src.data.schema import Claim, ClaimLabel, Evidence

LABELS: tuple[ClaimLabel, ...] = (
    ClaimLabel.SUPPORTS,
    ClaimLabel.REFUTES,
    ClaimLabel.NOT_ENOUGH_INFO,
    ClaimLabel.DISPUTED,
)
LABEL_TO_ID: dict[ClaimLabel, int] = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL: dict[int, ClaimLabel] = {i: label for label, i in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class JointExample:
    """One supervised example for 4-way claim classification."""

    claim_id: str
    claim_text: str
    evidence_text: str
    label: ClaimLabel

    @property
    def label_id(self) -> int:
        return LABEL_TO_ID[self.label]


def build_evidence_bundle(
    evidences: Sequence[Evidence],
    max_evidences: int = 3,
    separator: str = " [EVIDENCE] ",
) -> str:
    """Concatenate a small set of evidence passages into one text field."""
    if max_evidences <= 0:
        raise ValueError("max_evidences must be > 0")
    selected = [ev.text.strip() for ev in evidences[:max_evidences] if ev.text.strip()]
    if not selected:
        # The classifier should rarely see this because every prediction must
        # include evidence, but keeping a non-empty placeholder makes training
        # and tests robust to malformed examples.
        return "[NO_EVIDENCE]"
    return separator.join(selected)


def examples_from_gold_evidence(
    claims: Iterable[Claim],
    evidence_corpus: dict[str, Evidence],
    max_evidences: int = 3,
) -> list[JointExample]:
    """Build training examples using gold evidence ids from labeled claims.

    This is the cleanest classifier training signal: the model learns the label
    assuming the evidence set is correct. Pipeline evaluation still uses
    retrieved/reranked evidence, so this is not an oracle at test time.
    """
    examples: list[JointExample] = []
    for claim in claims:
        if claim.label is None:
            continue
        evidences = [
            evidence_corpus[ev_id]
            for ev_id in claim.evidence_ids
            if ev_id in evidence_corpus
        ]
        if not evidences:
            continue
        examples.append(
            JointExample(
                claim_id=claim.id,
                claim_text=claim.text,
                evidence_text=build_evidence_bundle(evidences, max_evidences=max_evidences),
                label=claim.label,
            )
        )
    return examples


def examples_from_evidence_batches(
    claims: Sequence[Claim],
    evidences_batch: Sequence[Sequence[Evidence]],
    max_evidences: int = 3,
) -> list[JointExample]:
    """Build examples from externally supplied evidence batches.

    Used for retrieved-evidence training, where the evidence batch comes from
    the current retriever rather than the gold annotations.
    """
    if len(claims) != len(evidences_batch):
        raise ValueError(
            f"claims and evidences_batch must have same length "
            f"({len(claims)} vs {len(evidences_batch)})"
        )
    examples: list[JointExample] = []
    for claim, evidences in zip(claims, evidences_batch):
        if claim.label is None:
            continue
        examples.append(
            JointExample(
                claim_id=claim.id,
                claim_text=claim.text,
                evidence_text=build_evidence_bundle(evidences, max_evidences=max_evidences),
                label=claim.label,
            )
        )
    return examples


def compute_balanced_class_weights(examples: Sequence[JointExample]) -> list[float]:
    """Return sklearn-style balanced weights without adding sklearn dependency."""
    if not examples:
        raise ValueError("examples must be non-empty")
    counts = {label: 0 for label in LABELS}
    for ex in examples:
        counts[ex.label] += 1

    total = len(examples)
    n_classes = len(LABELS)
    weights: list[float] = []
    for label in LABELS:
        count = counts[label]
        if count == 0:
            # Keep a finite weight. This should not happen for the full train
            # split, but tiny unit tests may omit classes.
            weights.append(0.0)
        else:
            weights.append(total / (n_classes * count))
    return weights
