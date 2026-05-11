"""Training data construction for the retrieval reranker.

The reranker is trained on (claim, evidence, label) pairs:
  - positives: gold evidence ids from train-claims.json
  - negatives: BM25-mined hard negatives not in the gold evidence set

This module is intentionally model-free. It only prepares examples, so it is
cheap to test and can be reused with different cross-encoder backbones.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Union

from src.data.schema import Claim, Evidence
from src.retrieval.base import Retriever

PathLike = Union[str, Path]


@dataclass(frozen=True)
class RerankExample:
    """A single pointwise reranker training example."""

    claim_id: str
    evidence_id: str
    claim_text: str
    evidence_text: str
    label: int  # 1 = relevant, 0 = non-relevant

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {self.label!r}")
        if not self.claim_id:
            raise ValueError("claim_id must be non-empty")
        if not self.evidence_id:
            raise ValueError("evidence_id must be non-empty")


def build_rerank_examples(
    claims: dict[str, Claim],
    evidence: dict[str, Evidence],
    retriever: Retriever,
    candidate_k: int = 500,
    negatives_per_positive: int = 4,
    seed: int = 42,
) -> list[RerankExample]:
    """Build hard-negative training pairs for the reranker.

    Args:
        claims: labeled training claims.
        evidence: full evidence corpus.
        retriever: first-stage retriever, normally BM25.
        candidate_k: number of first-stage candidates mined per claim.
        negatives_per_positive: how many negatives to sample for each positive.
        seed: deterministic sampling seed.

    Returns:
        Shuffled list of RerankExample objects.
    """
    if candidate_k <= 0:
        raise ValueError("candidate_k must be > 0")
    if negatives_per_positive < 0:
        raise ValueError("negatives_per_positive must be >= 0")

    labeled = [c for c in claims.values() if c.is_labeled and c.evidence_ids]
    if not labeled:
        raise ValueError("No labeled claims with gold evidences found")

    rng = random.Random(seed)
    examples: list[RerankExample] = []

    claim_texts = [c.text for c in labeled]
    retrieved_batch = retriever.retrieve_batch(claim_texts, candidate_k)

    for claim, retrieved in zip(labeled, retrieved_batch):
        gold_ids = tuple(eid for eid in claim.evidence_ids if eid in evidence)
        if not gold_ids:
            continue
        gold_set = set(gold_ids)

        # Positives: all gold evidences available in the corpus.
        for ev_id in gold_ids:
            examples.append(
                RerankExample(
                    claim_id=claim.id,
                    evidence_id=ev_id,
                    claim_text=claim.text,
                    evidence_text=evidence[ev_id].text,
                    label=1,
                )
            )

        # Negatives: BM25 candidates that are not gold. Preserve BM25 rank first,
        # then sample without replacement where possible.
        candidate_negative_ids: list[str] = []
        seen: set[str] = set()
        for result in retrieved:
            ev_id = result.evidence_id
            if ev_id in gold_set or ev_id in seen or ev_id not in evidence:
                continue
            candidate_negative_ids.append(ev_id)
            seen.add(ev_id)

        n_needed = negatives_per_positive * len(gold_ids)
        if n_needed == 0 or not candidate_negative_ids:
            continue

        if len(candidate_negative_ids) >= n_needed:
            sampled_negatives = rng.sample(candidate_negative_ids, n_needed)
        else:
            # In tiny test corpora there may be fewer candidates than requested.
            # Cycle deterministically after shuffling once.
            shuffled = candidate_negative_ids[:]
            rng.shuffle(shuffled)
            sampled_negatives = [shuffled[i % len(shuffled)] for i in range(n_needed)]

        for ev_id in sampled_negatives:
            examples.append(
                RerankExample(
                    claim_id=claim.id,
                    evidence_id=ev_id,
                    claim_text=claim.text,
                    evidence_text=evidence[ev_id].text,
                    label=0,
                )
            )

    rng.shuffle(examples)
    return examples


def save_rerank_examples(examples: Iterable[RerankExample], path: PathLike) -> None:
    """Write examples as JSONL for fast reuse."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=True) + "\n")


def load_rerank_examples(path: PathLike) -> list[RerankExample]:
    """Read examples written by save_rerank_examples."""
    path = Path(path)
    out: list[RerankExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(RerankExample(**json.loads(line)))
    return out
