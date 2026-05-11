"""Run BM25 + reranker + joint cross-encoder classifier."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.classification.joint import JointCrossEncoderClassifier
from src.config import CONFIG, get_device
from src.data.loader import load_claims, load_evidence
from src.pipeline import Pipeline
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.two_stage import TwoStageRetriever
from src.utils.io import write_predictions
from src.utils.logging_config import configure_logging, get_logger
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

LOGGER = get_logger("run_joint_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--candidate_k", type=int, default=500)
    parser.add_argument("--rerank_batch_size", type=int, default=32)
    parser.add_argument("--reranker_max_length", type=int, default=192)
    parser.add_argument(
        "--classifier_batch_size", type=int, default=32
        )  # currently stored on classifier call path only indirectly
    parser.add_argument("--classifier_max_length", type=int, default=256)
    parser.add_argument("--max_evidences", type=int, default=3)
    parser.add_argument("--reranker_dir", type=Path, default=CONFIG.paths.outputs / "reranker" / "model")
    parser.add_argument("--classifier_dir", type=Path, default=CONFIG.paths.outputs / "joint_classifier" / "model")
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    set_seed(args.seed)

    claims_path = CONFIG.paths.dev_claims if args.split == "dev" else CONFIG.paths.test_claims
    output_path = args.output or (
        CONFIG.paths.outputs / f"{args.split}-joint-predictions-k{args.top_k}.json"
    )

    LOGGER.info("Loading evidence corpus")
    evidence = load_evidence(CONFIG.paths.evidence)
    LOGGER.info("Loading BM25 index")
    bm25 = BM25Retriever.load(CONFIG.paths.outputs / "bm25_index")
    LOGGER.info("Loading reranker from %s", args.reranker_dir)
    reranker = CrossEncoderReranker.load(
        args.reranker_dir, device=get_device(), max_length=args.reranker_max_length
    )
    retriever = TwoStageRetriever(
        first_stage=bm25,
        reranker=reranker,
        evidence_corpus=evidence,
        candidate_k=args.candidate_k,
        rerank_batch_size=args.rerank_batch_size,
    )

    LOGGER.info("Loading joint classifier from %s", args.classifier_dir)
    classifier = JointCrossEncoderClassifier.load(
        args.classifier_dir,
        device=get_device(),
        max_length=args.classifier_max_length,
        max_evidences=args.max_evidences,
    )

    claims = load_claims(claims_path)
    LOGGER.info(
        "Running joint pipeline: split=%s, claims=%d, candidate_k=%d, top_k=%d",
        args.split,
        len(claims),
        args.candidate_k,
        args.top_k,
    )
    pipeline = Pipeline(
        retriever=retriever,
        classifier=classifier,
        evidence_corpus=evidence,
        top_k=args.top_k,
    )
    start = time.time()
    predictions = pipeline.predict_batch(list(claims.values()))
    elapsed = time.time() - start
    LOGGER.info("Completed in %.1fs (%.1fms/claim)", elapsed, 1000 * elapsed / len(claims))

    write_predictions(predictions, output_path)
    LOGGER.info("Predictions written to %s", output_path)
    if args.split == "dev":
        LOGGER.info(
            "Evaluate with: python eval.py --predictions %s --groundtruth %s",
            output_path,
            CONFIG.paths.dev_claims,
        )


if __name__ == "__main__":
    main()
