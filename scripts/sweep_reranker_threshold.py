"""Sweep dynamic-K thresholds for the BM25 + reranker pipeline on dev."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.classification.majority import MajorityClassifier
from src.config import CONFIG, get_device
from src.data.loader import load_claims, load_evidence
from src.data.schema import Prediction
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.two_stage import TwoStageRetriever
from src.utils.io import write_predictions
from src.utils.logging_config import configure_logging, get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

LOGGER = get_logger("sweep_reranker_threshold")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", type=str, default="-2,-1,0,1,2")
    parser.add_argument("--candidate_k", type=int, default=500)
    parser.add_argument("--min_k", type=int, default=1)
    parser.add_argument("--max_k", type=int, default=5)
    parser.add_argument("--rerank_batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument(
        "--model_dir", type=Path, default=CONFIG.paths.outputs / "reranker" / "model"
    )
    return parser.parse_args()


def safe_threshold_name(x: float) -> str:
    return str(x).replace("-", "m").replace(".", "p")


def main() -> None:
    args = parse_args()
    configure_logging()
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    evidence = load_evidence(CONFIG.paths.evidence)
    claims = load_claims(CONFIG.paths.dev_claims)
    train_claims = load_claims(CONFIG.paths.train_claims)
    classifier = MajorityClassifier.fit(train_claims)
    label = classifier.label

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

    claim_list = list(claims.values())
    claim_texts = [c.text for c in claim_list]
    LOGGER.info("Scoring candidates once for %d claims", len(claim_list))
    scored = retriever.score_candidates_batch(claim_texts)

    for threshold in thresholds:
        predictions: dict[str, Prediction] = {}
        for claim, items in zip(claim_list, scored):
            selected = [r for r in items[: args.max_k] if r.score >= threshold]
            if len(selected) < args.min_k:
                selected = items[: args.min_k]
            selected = selected[: args.max_k]
            predictions[claim.id] = Prediction(
                claim_id=claim.id,
                claim_text=claim.text,
                claim_label=label,
                evidence_ids=tuple(r.evidence_id for r in selected),
            )

        output_path = (
            CONFIG.paths.outputs
            / f"dev-reranker-predictions-threshold-{safe_threshold_name(threshold)}.json"
        )
        write_predictions(predictions, output_path)
        LOGGER.info("threshold=%.3f written to %s", threshold, output_path)
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
