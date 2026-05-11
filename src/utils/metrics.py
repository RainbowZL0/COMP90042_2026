"""Evaluation helpers mirroring eval.py without printing side effects."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np

PathLike = Union[str, Path]


@dataclass(frozen=True)
class EvalMetrics:
    evidence_f: float
    accuracy: float
    harmonic_mean: float
    n_claims: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "evidence_f": self.evidence_f,
            "accuracy": self.accuracy,
            "harmonic_mean": self.harmonic_mean,
            "n_claims": self.n_claims,
        }


def load_json(path: PathLike) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_eval_metrics(predictions: dict, groundtruth: dict) -> EvalMetrics:
    """Compute the same aggregate metrics as the official eval.py."""
    f_scores: list[float] = []
    acc_scores: list[float] = []

    for claim_id, claim in sorted(groundtruth.items()):
        if (
            claim_id not in predictions
            or "claim_label" not in predictions[claim_id]
            or "evidences" not in predictions[claim_id]
        ):
            continue

        pred = predictions[claim_id]
        acc_scores.append(1.0 if pred["claim_label"] == claim["claim_label"] else 0.0)

        evidence_f = 0.0
        pred_evidences = pred["evidences"]
        if isinstance(pred_evidences, list) and len(pred_evidences) > 0:
            pred_set = set(pred_evidences)
            correct = sum(1 for ev_id in claim["evidences"] if ev_id in pred_set)
            if correct > 0:
                recall = float(correct) / len(claim["evidences"])
                precision = float(correct) / len(pred_evidences)
                evidence_f = (2 * precision * recall) / (precision + recall)
        f_scores.append(evidence_f)

    mean_f = float(np.mean(f_scores if f_scores else [0.0]))
    mean_acc = float(np.mean(acc_scores if acc_scores else [0.0]))
    hmean = 0.0 if mean_f == 0.0 and mean_acc == 0.0 else float((2 * mean_f * mean_acc) / (mean_f + mean_acc))
    return EvalMetrics(mean_f, mean_acc, hmean, len(acc_scores))


def evaluate_prediction_file(predictions_path: PathLike, groundtruth_path: PathLike) -> EvalMetrics:
    return compute_eval_metrics(load_json(predictions_path), load_json(groundtruth_path))
