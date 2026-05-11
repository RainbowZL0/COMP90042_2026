"""Sweep fixed output K for the BM25 + reranker pipeline on dev."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.classification.majority import MajorityClassifier
from src.config import CONFIG, get_device
from src.data.loader import load_claims, load_evidence
from src.pipeline import Pipeline
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.two_stage import TwoStageRetriever
from src.utils.io import write_predictions
from src.utils.logging_config import configure_logging, get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

LOGGER = get_logger("sweep_reranker_k")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ks", type=str, default="2,3,4,5")
    parser.add_argument("--candidate_k", type=int, default=500)
    parser.add_argument("--rerank_batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument(
        "--model_dir", type=Path, default=CONFIG.paths.outputs / "reranker" / "model"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    ks = [int(x) for x in args.ks.split(",") if x.strip()]

    evidence = load_evidence(CONFIG.paths.evidence)
    claims = load_claims(CONFIG.paths.dev_claims)
    train_claims = load_claims(CONFIG.paths.train_claims)
    classifier = MajorityClassifier.fit(train_claims)

    bm25 = BM25Retriever.load(CONFIG.paths.outputs / "bm25_index")
    reranker = CrossEncoderReranker.load(
        args.model_dir, device=get_device(), max_length=args.max_length
    )
    retriever = TwoStageRetriever(
        first_stage=bm25,
        reranker=reranker,
        evidence_corpus=evidence,
        candidate_k=args.candidate_k,
        rerank_batch_size=args.rerank_batch_size,
    )

    LOGGER.info("Sweeping fixed K values: %s", ks)
    for k in ks:
        pipeline = Pipeline(retriever, classifier, evidence, top_k=k)
        predictions = pipeline.predict_batch(list(claims.values()))
        output_path = CONFIG.paths.outputs / f"dev-reranker-predictions-k{k}.json"
        write_predictions(predictions, output_path)
        LOGGER.info("K=%d written to %s", k, output_path)
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "eval.py"),
                "--predictions",
                str(output_path),
                "--groundtruth",
                str(CONFIG.paths.dev_claims),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
