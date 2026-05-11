"""Evaluate stance-attention classifier with gold evidence only.

This is a classifier diagnostic, not an end-to-end system score.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.joint_data import LABELS
from src.classification.stance_attention import StanceAttentionClassifier
from src.classification.stance_attention_data import examples_from_gold_evidence
from src.config import CONFIG, get_device
from src.data.loader import load_claims, load_evidence
from src.utils.logging_config import configure_logging, get_logger
from src.utils.mlflow_utils import mlflow_run

LOGGER = get_logger("eval_stance_attention_gold")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "dev"], default="dev")
    parser.add_argument(
        "--classifier_dir", type=Path, default=CONFIG.paths.outputs / "stance_attention_classifier" / "model"
    )
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--max_evidences", type=int, default=3)
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--mlflow_experiment", default="COMP90042")
    parser.add_argument("--mlflow_tracking_uri", default=CONFIG.paths.outputs / "mlruns")
    parser.add_argument("--run_name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    claims_path = CONFIG.paths.train_claims if args.split == "train" else CONFIG.paths.dev_claims
    claims = load_claims(claims_path)
    evidence = load_evidence(CONFIG.paths.evidence)
    examples = examples_from_gold_evidence(list(claims.values()), evidence, max_evidences=args.max_evidences)
    classifier = StanceAttentionClassifier.load(
        args.classifier_dir,
        device=get_device(),
        max_length=args.max_length,
        max_evidences=args.max_evidences,
    )
    preds = classifier.classify_batch(
        [ex.claim_text for ex in examples],
        [[type("E", (), {"id": f"tmp-{i}", "text": text})() for text in ex.evidence_texts if text] for i, ex in
         enumerate(examples)],
    )
    total = len(examples)
    correct = sum(1 for ex, pred in zip(examples, preds) if ex.label == pred)
    acc = correct / max(total, 1)
    LOGGER.info("Gold-evidence %s accuracy = %.4f (%d/%d)", args.split, acc, correct, total)

    by_label_total = Counter(ex.label for ex in examples)
    by_label_correct = Counter(ex.label for ex, pred in zip(examples, preds) if ex.label == pred)
    confusion = defaultdict(Counter)
    for ex, pred in zip(examples, preds):
        confusion[ex.label][pred] += 1

    LOGGER.info("Per-label accuracy:")
    metrics = {"gold_accuracy": acc}
    for label in LABELS:
        label_acc = by_label_correct[label] / max(by_label_total[label], 1)
        metrics[f"gold_acc_{label.value}"] = label_acc
        LOGGER.info("  %-16s %.4f (%d/%d)", label.value, label_acc, by_label_correct[label], by_label_total[label])
    LOGGER.info("Confusion rows=gold cols=pred:")
    for label in LABELS:
        LOGGER.info("  %s -> %s", label.value, dict(confusion[label]))

    with mlflow_run(
        enabled=args.mlflow,
        experiment_name=args.mlflow_experiment,
        run_name=args.run_name or f"stance_attention_gold_eval_{args.split}",
        tracking_uri=args.mlflow_tracking_uri,
        tags={"stage": "diagnostic", "component": "stance_attention_classifier", "split": args.split},
    ) as run:
        run.log_params(vars(args))
        run.log_metrics(metrics)


if __name__ == "__main__":
    main()
