"""Train the Phase 4B per-evidence stance + attention classifier."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.joint_data import LABELS
from src.classification.stance_attention import (
    StanceAttentionClassifier,
    train_stance_attention_classifier,
)
from src.classification.stance_attention_data import (
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
from src.utils.mlflow_utils import mlflow_run
from src.utils.seed import set_seed

LOGGER = get_logger("train_stance_attention_classifier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--evidence_source", choices=["gold", "retrieved"], default="retrieved")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--candidate_k", type=int, default=500)
    parser.add_argument("--reranker_dir", type=Path, default=CONFIG.paths.outputs / "reranker" / "model")
    parser.add_argument("--rerank_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--max_evidences", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--stance_loss_weight", type=float, default=0.3)
    parser.add_argument("--dev_fraction", type=float, default=0.15)
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    parser.add_argument(
        "--output_dir", type=Path, default=CONFIG.paths.outputs / "stance_attention_classifier" / "model"
    )
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--mlflow_experiment", default="COMP90042")
    parser.add_argument("--mlflow_tracking_uri", default=CONFIG.paths.outputs / "mlruns")
    parser.add_argument("--run_name", default=None)
    return parser.parse_args()


def split_claims_by_id(claim_ids: list[str], dev_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    if not 0.0 <= dev_fraction < 1.0:
        raise ValueError("dev_fraction must be in [0, 1)")
    rng = random.Random(seed)
    ids = list(claim_ids)
    rng.shuffle(ids)
    n_dev = int(round(len(ids) * dev_fraction))
    return set(ids[n_dev:]), set(ids[:n_dev])


def retrieved_evidence_batches(claims, evidence: dict[str, Evidence], args: argparse.Namespace) -> list[list[Evidence]]:
    LOGGER.info("Loading BM25 and reranker to build retrieved-evidence training examples")
    bm25 = BM25Retriever.load(CONFIG.paths.outputs / "bm25_index")
    reranker = CrossEncoderReranker.load(args.reranker_dir, device=get_device(), max_length=192)
    retriever = TwoStageRetriever(
        first_stage=bm25,
        reranker=reranker,
        evidence_corpus=evidence,
        candidate_k=args.candidate_k,
        rerank_batch_size=args.rerank_batch_size,
    )
    results = retriever.retrieve_batch([c.text for c in claims], top_k=args.top_k)
    return [[evidence[r.evidence_id] for r in items if r.evidence_id in evidence] for items in results]


def main() -> None:
    args = parse_args()
    configure_logging()
    set_seed(args.seed)

    with mlflow_run(
        enabled=args.mlflow,
        experiment_name=args.mlflow_experiment,
        run_name=args.run_name or f"stance_attention_train_{args.evidence_source}_k{args.top_k}",
        tracking_uri=args.mlflow_tracking_uri,
        tags={"stage": "train", "component": "stance_attention_classifier"},
    ) as run:
        run.log_params(vars(args))

        LOGGER.info("Device = %s", get_device())
        LOGGER.info("Loading train claims and evidence corpus")
        train_claims = load_claims(CONFIG.paths.train_claims)
        evidence = load_evidence(CONFIG.paths.evidence)
        labeled_claims = [c for c in train_claims.values() if c.label is not None]
        train_ids, dev_ids = split_claims_by_id([c.id for c in labeled_claims], args.dev_fraction, args.seed)
        train_claim_list = [c for c in labeled_claims if c.id in train_ids]
        dev_claim_list = [c for c in labeled_claims if c.id in dev_ids]
        LOGGER.info("Claim split: %d train claims, %d internal-dev claims", len(train_claim_list), len(dev_claim_list))
        LOGGER.info("Evidence source = %s", args.evidence_source)

        if args.evidence_source == "gold":
            train_examples = examples_from_gold_evidence(train_claim_list, evidence, max_evidences=args.max_evidences)
            dev_examples = examples_from_gold_evidence(dev_claim_list, evidence, max_evidences=args.max_evidences)
        else:
            train_batches = retrieved_evidence_batches(train_claim_list, evidence, args)
            dev_batches = retrieved_evidence_batches(dev_claim_list, evidence, args)
            train_examples = examples_from_evidence_batches(
                train_claim_list, train_batches, max_evidences=args.max_evidences
            )
            dev_examples = examples_from_evidence_batches(dev_claim_list, dev_batches, max_evidences=args.max_evidences)

        LOGGER.info("Examples: %d train, %d internal-dev", len(train_examples), len(dev_examples))
        counts = {label.value: Counter(ex.label for ex in train_examples)[label] for label in LABELS}
        LOGGER.info("Train label counts: %s", counts)
        run.log_params({f"train_count_{k}": v for k, v in counts.items()})

        if args.no_class_weights:
            class_weights = None
            LOGGER.info("Class weights disabled")
        else:
            class_weights = compute_balanced_class_weights(train_examples)
            LOGGER.info("Class weights: %s", {label.value: round(w, 4) for label, w in zip(LABELS, class_weights)})

        LOGGER.info("Loading model: %s", args.model_name)
        classifier = StanceAttentionClassifier.from_pretrained_base(
            args.model_name,
            device=get_device(),
            max_length=args.max_length,
            max_evidences=args.max_evidences,
            dropout=args.dropout,
        )
        history = train_stance_attention_classifier(
            classifier,
            train_examples=train_examples,
            dev_examples=dev_examples if dev_examples else None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            class_weights=class_weights,
            stance_loss_weight=args.stance_loss_weight,
        )
        for i, loss in enumerate(history.get("train_loss", []), start=1):
            run.log_metrics({"train_loss": loss}, step=i)
        for i, acc in enumerate(history.get("dev_accuracy", []), start=1):
            run.log_metrics({"internal_dev_accuracy": acc}, step=i)

        LOGGER.info("Saving classifier to %s", args.output_dir)
        classifier.save(args.output_dir, dropout=args.dropout)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        history_path = args.output_dir / "training_history.json"
        args_path = args.output_dir / "training_args.json"
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=True, indent=2)
            f.write("\n")
        with args_path.open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=True, indent=2, default=str)
            f.write("\n")
        run.log_artifact(history_path)
        run.log_artifact(args_path)
        LOGGER.info("Done")


if __name__ == "__main__":
    main()
