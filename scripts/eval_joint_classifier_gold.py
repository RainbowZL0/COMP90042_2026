"""Evaluate the joint classifier with gold dev evidence only.

This is not the official pipeline score. It is a diagnostic upper-bound-ish
check for the classifier: if accuracy is poor even with gold evidence, the
classifier itself is the bottleneck; if it is good here but poor in the full
pipeline, retrieval noise is the bottleneck.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from src.classification.joint import JointCrossEncoderClassifier
from src.config import CONFIG, get_device
from src.data.loader import load_claims, load_evidence
from src.data.schema import ClaimLabel
from src.utils.logging_config import configure_logging, get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


LOGGER = get_logger("eval_joint_classifier_gold")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "dev"], default="dev")
    parser.add_argument("--classifier_dir", type=Path, default=CONFIG.paths.outputs / "joint_classifier" / "model")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_evidences", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    claims_path = CONFIG.paths.train_claims if args.split == "train" else CONFIG.paths.dev_claims
    claims = load_claims(claims_path)
    evidence = load_evidence(CONFIG.paths.evidence)
    classifier = JointCrossEncoderClassifier.load(
        args.classifier_dir,
        device=get_device(),
        max_length=args.max_length,
        max_evidences=args.max_evidences,
    )

    labeled = [c for c in claims.values() if c.label is not None]
    claim_texts = [c.text for c in labeled]
    evidences_batch = [
        [evidence[ev_id] for ev_id in c.evidence_ids if ev_id in evidence]
        for c in labeled
    ]
    preds = classifier.classify_batch(claim_texts, evidences_batch)

    total = len(labeled)
    correct = sum(1 for c, p in zip(labeled, preds) if c.label == p)
    LOGGER.info("Gold-evidence %s accuracy = %.4f (%d/%d)", args.split, correct / total, correct, total)

    per_label_total = Counter(c.label for c in labeled)
    per_label_correct = Counter(c.label for c, p in zip(labeled, preds) if c.label == p)
    LOGGER.info("Per-label accuracy:")
    for label in ClaimLabel:
        denom = per_label_total[label]
        acc = per_label_correct[label] / denom if denom else 0.0
        LOGGER.info("  %-16s %.4f (%d/%d)", label.value, acc, per_label_correct[label], denom)

    confusion: dict[ClaimLabel, Counter] = defaultdict(Counter)
    for c, p in zip(labeled, preds):
        confusion[c.label][p] += 1
    LOGGER.info("Confusion rows=gold cols=pred:")
    for gold in ClaimLabel:
        LOGGER.info("  %s -> %s", gold.value, dict(confusion[gold]))


if __name__ == "__main__":
    main()
