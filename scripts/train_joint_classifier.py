"""Train the Phase 4 joint evidence classifier.

Default mode trains on gold evidence because it gives the cleanest label signal.
An optional retrieved-evidence mode is provided to match pipeline inference more
closely after the first model is working.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.joint import JointCrossEncoderClassifier, train_joint_classifier
from src.classification.joint_data import (
    LABELS,
    compute_balanced_class_weights,
    examples_from_evidence_batches,
    examples_from_gold_evidence,
)
from src.config import CONFIG, get_device
from src.data.loader import load_claims, load_evidence
from src.data.schema import Evidence
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.two_stage import TwoStageRetriever
from src.utils.logging_config import configure_logging, get_logger
from src.utils.seed import set_seed

LOGGER = get_logger("train_joint_classifier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_name",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="HF checkpoint used to initialise the classifier head.",
    )
    parser.add_argument("--evidence_source", choices=["gold", "retrieved"], default="gold")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--candidate_k", type=int, default=500)
    parser.add_argument("--reranker_dir", type=Path, default=CONFIG.paths.outputs / "reranker" / "model")
    parser.add_argument("--rerank_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_evidences", type=int, default=3)
    parser.add_argument("--dev_fraction", type=float, default=0.15)
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    parser.add_argument("--output_dir", type=Path, default=CONFIG.paths.outputs / "joint_classifier" / "model")
    return parser.parse_args()


def split_claims_by_id(claim_ids: list[str], dev_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    if not 0.0 <= dev_fraction < 1.0:
        raise ValueError("dev_fraction must be in [0, 1)")
    rng = random.Random(seed)
    ids = list(claim_ids)
    rng.shuffle(ids)
    n_dev = int(round(len(ids) * dev_fraction))
    dev_ids = set(ids[:n_dev])
    train_ids = set(ids[n_dev:])
    return train_ids, dev_ids


def retrieved_evidence_batches(
    claims,
    evidence: dict[str, Evidence],
    args: argparse.Namespace,
) -> list[list[Evidence]]:
    LOGGER.info("Loading BM25 and reranker to build retrieved-evidence training examples")
    bm25 = BM25Retriever.load(CONFIG.paths.outputs / "bm25_index")
    reranker = CrossEncoderReranker.load(
        args.reranker_dir, device=get_device(), max_length=192
    )
    retriever = TwoStageRetriever(
        first_stage=bm25,
        reranker=reranker,
        evidence_corpus=evidence,
        candidate_k=args.candidate_k,
        rerank_batch_size=args.rerank_batch_size,
    )
    results = retriever.retrieve_batch([c.text for c in claims], top_k=args.top_k)
    batches: list[list[Evidence]] = []
    for items in results:
        evs = [evidence[r.evidence_id] for r in items if r.evidence_id in evidence]
        batches.append(evs)
    return batches


def main() -> None:
    args = parse_args()
    configure_logging()
    set_seed(args.seed)

    LOGGER.info("Device = %s", get_device())
    LOGGER.info("Loading train claims and evidence corpus")
    train_claims = load_claims(CONFIG.paths.train_claims)
    evidence = load_evidence(CONFIG.paths.evidence)

    labeled_claims = [c for c in train_claims.values() if c.label is not None]
    train_ids, dev_ids = split_claims_by_id([c.id for c in labeled_claims], args.dev_fraction, args.seed)
    train_claim_list = [c for c in labeled_claims if c.id in train_ids]
    dev_claim_list = [c for c in labeled_claims if c.id in dev_ids]

    LOGGER.info(
        "Claim split: %d train claims, %d internal-dev claims",
        len(train_claim_list),
        len(dev_claim_list),
    )
    LOGGER.info("Evidence source = %s", args.evidence_source)

    if args.evidence_source == "gold":
        train_examples = examples_from_gold_evidence(
            train_claim_list, evidence, max_evidences=args.max_evidences
        )
        dev_examples = examples_from_gold_evidence(
            dev_claim_list, evidence, max_evidences=args.max_evidences
        )
    else:
        train_batches = retrieved_evidence_batches(train_claim_list, evidence, args)
        dev_batches = retrieved_evidence_batches(dev_claim_list, evidence, args)
        train_examples = examples_from_evidence_batches(
            train_claim_list, train_batches, max_evidences=args.max_evidences
        )
        dev_examples = examples_from_evidence_batches(
            dev_claim_list, dev_batches, max_evidences=args.max_evidences
        )

    LOGGER.info("Examples: %d train, %d internal-dev", len(train_examples), len(dev_examples))
    LOGGER.info(
        "Train label counts: %s",
        {label.value: Counter(ex.label for ex in train_examples)[label] for label in LABELS},
    )

    if args.no_class_weights:
        class_weights = None
        LOGGER.info("Class weights disabled")
    else:
        class_weights = compute_balanced_class_weights(train_examples)
        LOGGER.info(
            "Class weights: %s",
            {label.value: round(w, 4) for label, w in zip(LABELS, class_weights)},
        )

    LOGGER.info("Loading model: %s", args.model_name)
    classifier = JointCrossEncoderClassifier.from_pretrained_base(
        args.model_name,
        device=get_device(),
        max_length=args.max_length,
        max_evidences=args.max_evidences,
    )
    history = train_joint_classifier(
        classifier,
        train_examples=train_examples,
        dev_examples=dev_examples if dev_examples else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
    )

    LOGGER.info("Saving classifier to %s", args.output_dir)
    classifier.save(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=True, indent=2)
        f.write("\n")
    with (args.output_dir / "training_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=True, indent=2, default=str)
        f.write("\n")
    LOGGER.info("Done")


if __name__ == "__main__":
    main()
