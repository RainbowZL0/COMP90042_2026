"""Run BM25 + reranker + stance-attention classifier."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.stance_attention import StanceAttentionClassifier
from src.config import CONFIG, get_device
from src.data.loader import load_claims, load_evidence
from src.pipeline import Pipeline
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.two_stage import TwoStageRetriever
from src.utils.io import write_predictions
from src.utils.logging_config import configure_logging, get_logger
from src.utils.metrics import evaluate_prediction_file
from src.utils.mlflow_utils import mlflow_run
from src.utils.seed import set_seed

LOGGER = get_logger("run_stance_attention_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--candidate_k", type=int, default=500)
    parser.add_argument("--rerank_batch_size", type=int, default=32)
    parser.add_argument("--reranker_max_length", type=int, default=192)
    parser.add_argument("--classifier_max_length", type=int, default=192)
    parser.add_argument("--max_evidences", type=int, default=3)
    parser.add_argument("--reranker_dir", type=Path, default=CONFIG.paths.outputs / "reranker" / "model")
    parser.add_argument(
        "--classifier_dir", type=Path, default=CONFIG.paths.outputs / "stance_attention_classifier" / "model"
    )
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--mlflow_experiment", default="COMP90042")
    parser.add_argument("--mlflow_tracking_uri", default=CONFIG.paths.outputs / "mlruns")
    parser.add_argument("--run_name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    set_seed(args.seed)

    claims_path = CONFIG.paths.dev_claims if args.split == "dev" else CONFIG.paths.test_claims
    output_path = args.output or CONFIG.paths.outputs / f"{args.split}-stance-attention-predictions-k{args.top_k}.json"

    with mlflow_run(
        enabled=args.mlflow,
        experiment_name=args.mlflow_experiment,
        run_name=args.run_name or f"stance_attention_pipeline_{args.split}_k{args.top_k}",
        tracking_uri=args.mlflow_tracking_uri,
        tags={"stage": "pipeline", "component": "stance_attention_classifier", "split": args.split},
    ) as run:
        run.log_params(vars(args))

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
        LOGGER.info("Loading stance-attention classifier from %s", args.classifier_dir)
        classifier = StanceAttentionClassifier.load(
            args.classifier_dir,
            device=get_device(),
            max_length=args.classifier_max_length,
            max_evidences=args.max_evidences,
        )
        claims = load_claims(claims_path)
        LOGGER.info(
            "Running stance-attention pipeline: split=%s, claims=%d, candidate_k=%d, top_k=%d", args.split, len(claims),
            args.candidate_k, args.top_k
        )
        pipeline = Pipeline(retriever=retriever, classifier=classifier, evidence_corpus=evidence, top_k=args.top_k)
        start = time.time()
        predictions = pipeline.predict_batch(list(claims.values()))
        elapsed = time.time() - start
        LOGGER.info("Completed in %.1fs (%.1fms/claim)", elapsed, 1000 * elapsed / len(claims))
        write_predictions(predictions, output_path)
        LOGGER.info("Predictions written to %s", output_path)
        run.log_artifact(output_path)
        run.log_metrics({"elapsed_seconds": elapsed, "ms_per_claim": 1000 * elapsed / len(claims)})
        if args.split == "dev":
            metrics = evaluate_prediction_file(output_path, CONFIG.paths.dev_claims)
            LOGGER.info(
                "Dev metrics: F=%.4f A=%.4f H=%.4f", metrics.evidence_f, metrics.accuracy, metrics.harmonic_mean
            )
            run.log_metrics(metrics.as_dict())


if __name__ == "__main__":
    main()
