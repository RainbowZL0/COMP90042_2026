"""Evaluate a prediction file and optionally log the result to MLflow."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG
from src.utils.metrics import evaluate_prediction_file
from src.utils.mlflow_utils import mlflow_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--groundtruth", default=CONFIG.paths.dev_claims, type=Path)
    parser.add_argument("--system_name", default="manual_eval")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--mlflow_experiment", default="COMP90042")
    parser.add_argument("--mlflow_tracking_uri", default=CONFIG.paths.outputs / "mlruns")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_prediction_file(args.predictions, args.groundtruth)
    print("Evidence Retrieval F-score (F)    =", metrics.evidence_f)
    print("Claim Classification Accuracy (A) =", metrics.accuracy)
    print("Harmonic Mean of F and A          =", metrics.harmonic_mean)

    with mlflow_run(
        enabled=args.mlflow,
        experiment_name=args.mlflow_experiment,
        run_name=args.run_name or args.system_name,
        tracking_uri=args.mlflow_tracking_uri,
        tags={"stage": "eval", "system": args.system_name},
    ) as run:
        run.log_params(
            {
                "system_name": args.system_name,
                "predictions": args.predictions,
                "groundtruth": args.groundtruth,
            }
        )
        run.log_metrics(metrics.as_dict())
        run.log_artifact(args.predictions)


if __name__ == "__main__":
    main()
