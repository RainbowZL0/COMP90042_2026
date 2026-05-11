"""Train the cross-encoder evidence reranker.

Example:
    uv run python scripts/train_reranker.py \
      --candidate_k 500 \
      --negatives_per_positive 4 \
      --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
      --epochs 3 \
      --batch_size 32
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from src.config import CONFIG, get_device
from src.data.loader import load_claims, load_evidence
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.rerank_data import (
    RerankExample,
    build_rerank_examples,
    load_rerank_examples,
    save_rerank_examples,
)
from src.retrieval.reranker import CrossEncoderReranker, train_reranker
from src.utils.logging_config import configure_logging, get_logger
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

LOGGER = get_logger("train_reranker")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_k", type=int, default=500)
    parser.add_argument("--negatives_per_positive", type=int, default=4)
    parser.add_argument(
        "--model_name",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="HF model id or local path. A small cross-encoder keeps Colab feasible.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--dev_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    parser.add_argument(
        "--examples_path",
        type=Path,
        default=CONFIG.paths.outputs / "reranker" / "train_pairs.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=CONFIG.paths.outputs / "reranker" / "model",
    )
    parser.add_argument(
        "--force_rebuild_examples",
        action="store_true",
        help="Ignore cached train_pairs.jsonl and rebuild BM25 hard negatives.",
    )
    return parser.parse_args()


def split_examples(
    examples: list[RerankExample], dev_fraction: float, seed: int
) -> tuple[list[RerankExample], list[RerankExample]]:
    if not 0 <= dev_fraction < 1:
        raise ValueError("dev_fraction must be in [0, 1)")
    shuffled = examples[:]
    random.Random(seed).shuffle(shuffled)
    n_dev = int(len(shuffled) * dev_fraction)
    return shuffled[n_dev:], shuffled[:n_dev]


def main() -> None:
    args = parse_args()
    configure_logging()
    set_seed(args.seed)

    LOGGER.info("Device = %s", get_device())
    LOGGER.info("Loading or building reranker examples")
    if args.examples_path.exists() and not args.force_rebuild_examples:
        examples = load_rerank_examples(args.examples_path)
        LOGGER.info("Loaded %d cached examples from %s", len(examples), args.examples_path)
    else:
        LOGGER.info("Loading train claims from %s", CONFIG.paths.train_claims)
        train_claims = load_claims(CONFIG.paths.train_claims)
        LOGGER.info("Loading evidence corpus from %s", CONFIG.paths.evidence)
        evidence = load_evidence(CONFIG.paths.evidence)
        LOGGER.info("Loading BM25 index")
        bm25 = BM25Retriever.load(CONFIG.paths.outputs / "bm25_index")
        LOGGER.info(
            "Mining hard negatives: candidate_k=%d, negatives_per_positive=%d",
            args.candidate_k,
            args.negatives_per_positive,
        )
        examples = build_rerank_examples(
            train_claims,
            evidence,
            bm25,
            candidate_k=args.candidate_k,
            negatives_per_positive=args.negatives_per_positive,
            seed=args.seed,
        )
        save_rerank_examples(examples, args.examples_path)
        LOGGER.info("Saved %d examples to %s", len(examples), args.examples_path)

    positives = sum(ex.label == 1 for ex in examples)
    negatives = len(examples) - positives
    LOGGER.info("Examples: %d positive, %d negative", positives, negatives)

    train_examples, dev_examples = split_examples(
        examples, dev_fraction=args.dev_fraction, seed=args.seed
    )
    LOGGER.info(
        "Train/dev split: %d train, %d dev", len(train_examples), len(dev_examples)
    )

    LOGGER.info("Loading model: %s", args.model_name)
    reranker = CrossEncoderReranker.from_pretrained(
        args.model_name, device=get_device(), max_length=args.max_length
    )

    history = train_reranker(
        reranker,
        train_examples=train_examples,
        dev_examples=dev_examples if dev_examples else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    LOGGER.info("Saving model to %s", args.output_dir)
    reranker.save(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=True, indent=2)
    LOGGER.info("Done")


if __name__ == "__main__":
    main()
