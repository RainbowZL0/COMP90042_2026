"""Data helpers for per-evidence stance + attention verification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.classification.joint_data import LABELS, LABEL_TO_ID
from src.data.schema import Claim, ClaimLabel, Evidence

STANCE_SUPPORT = 0
STANCE_REFUTE = 1
STANCE_NEUTRAL = 2
STANCE_LABELS = ("SUPPORT", "REFUTE", "NEUTRAL")


@dataclass(frozen=True)
class StanceAttentionExample:
    """One claim-level training instance with a fixed-size evidence bundle."""

    claim_id: str
    claim_text: str
    evidence_texts: tuple[str, ...]
    evidence_mask: tuple[int, ...]
    label: ClaimLabel
    stance_labels: tuple[int, ...]
    stance_mask: tuple[int, ...]

    @property
    def label_id(self) -> int:
        return LABEL_TO_ID[self.label]


def _pad_evidences(evidences: Sequence[Evidence], max_evidences: int) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if max_evidences <= 0:
        raise ValueError("max_evidences must be > 0")
    selected = list(evidences[:max_evidences])
    texts = [ev.text for ev in selected]
    mask = [1] * len(texts)
    while len(texts) < max_evidences:
        texts.append("")
        mask.append(0)
    return tuple(texts), tuple(mask)


def _gold_stance_for_claim(label: ClaimLabel) -> tuple[int, int] | None:
    """Return (stance_label, mask_value) for gold SUPPORTS/REFUTES only."""
    if label == ClaimLabel.SUPPORTS:
        return STANCE_SUPPORT, 1
    if label == ClaimLabel.REFUTES:
        return STANCE_REFUTE, 1
    return None


def example_from_evidences(
    claim: Claim,
    evidences: Sequence[Evidence],
    max_evidences: int = 3,
    gold_evidence_ids: set[str] | None = None,
) -> StanceAttentionExample | None:
    """Build one example.

    Auxiliary stance labels are deliberately weak supervision:
    - SUPPORTS gold evidences inherit SUPPORT.
    - REFUTES gold evidences inherit REFUTE.
    - DISPUTED/NEI and non-gold retrieved evidences do not receive stance loss.
    """
    if claim.label is None:
        return None

    texts, evidence_mask = _pad_evidences(evidences, max_evidences=max_evidences)
    selected = list(evidences[:max_evidences])
    weak = _gold_stance_for_claim(claim.label)

    stance_labels: list[int] = []
    stance_mask: list[int] = []
    for ev in selected:
        if weak is not None and (gold_evidence_ids is None or ev.id in gold_evidence_ids):
            stance_labels.append(weak[0])
            stance_mask.append(1)
        else:
            stance_labels.append(STANCE_NEUTRAL)
            stance_mask.append(0)

    while len(stance_labels) < max_evidences:
        stance_labels.append(STANCE_NEUTRAL)
        stance_mask.append(0)

    return StanceAttentionExample(
        claim_id=claim.id,
        claim_text=claim.text,
        evidence_texts=texts,
        evidence_mask=evidence_mask,
        label=claim.label,
        stance_labels=tuple(stance_labels),
        stance_mask=tuple(stance_mask),
    )


def examples_from_gold_evidence(
    claims: Sequence[Claim],
    evidence_corpus: dict[str, Evidence],
    max_evidences: int = 3,
) -> list[StanceAttentionExample]:
    examples: list[StanceAttentionExample] = []
    for claim in claims:
        if claim.label is None:
            continue
        evidences = [evidence_corpus[ev_id] for ev_id in claim.evidence_ids if ev_id in evidence_corpus]
        if not evidences:
            continue
        ex = example_from_evidences(
            claim,
            evidences,
            max_evidences=max_evidences,
            gold_evidence_ids=set(claim.evidence_ids),
        )
        if ex is not None:
            examples.append(ex)
    return examples


def examples_from_evidence_batches(
    claims: Sequence[Claim],
    evidences_batch: Sequence[Sequence[Evidence]],
    max_evidences: int = 3,
) -> list[StanceAttentionExample]:
    if len(claims) != len(evidences_batch):
        raise ValueError(
            f"claims and evidences_batch must have same length ({len(claims)} vs {len(evidences_batch)})"
        )
    examples: list[StanceAttentionExample] = []
    for claim, evidences in zip(claims, evidences_batch):
        if claim.label is None:
            continue
        ex = example_from_evidences(
            claim,
            evidences,
            max_evidences=max_evidences,
            gold_evidence_ids=set(claim.evidence_ids),
        )
        if ex is not None:
            examples.append(ex)
    return examples


def compute_balanced_class_weights(examples: Sequence[StanceAttentionExample]) -> list[float]:
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
        weights.append(0.0 if count == 0 else total / (n_classes * count))
    return weights
