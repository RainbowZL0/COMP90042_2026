"""Read/write prediction JSON files.

The output format is dictated by `eval.py` (see project root). It reads:

    predictions[claim_id]["claim_label"]  -> str  (must be one of 4 labels)
    predictions[claim_id]["evidences"]    -> list[str]

We additionally include "claim_text" because the baseline file uses it and
having it makes manual inspection of predictions far easier. eval.py
ignores any extra keys.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from src.data.schema import ClaimLabel, Prediction

PathLike = Union[str, Path]


def write_predictions(predictions: dict[str, Prediction], path: PathLike) -> None:
    """Write predictions to a JSON file in eval.py's expected format.

    Creates parent directories if they don't exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out: dict[str, dict] = {}
    for claim_id, pred in predictions.items():
        if claim_id != pred.claim_id:
            raise ValueError(
                f"Inconsistent claim id: dict key {claim_id!r} "
                f"vs Prediction.claim_id {pred.claim_id!r}"
            )
        out[claim_id] = {
            "claim_text": pred.claim_text,
            "claim_label": pred.claim_label.value,
            "evidences": list(pred.evidence_ids),
        }

    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=True, indent=2)


def read_predictions(path: PathLike) -> dict[str, Prediction]:
    """Read predictions from a JSON file. Inverse of write_predictions."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    out: dict[str, Prediction] = {}
    for claim_id, fields in raw.items():
        out[claim_id] = Prediction(
            claim_id=claim_id,
            claim_text=fields.get("claim_text", ""),
            claim_label=ClaimLabel(fields["claim_label"]),
            evidence_ids=tuple(fields["evidences"]),
        )
    return out
