"""Load claims and the evidence corpus from JSON files.

Format spec (from assignment):

    Labeled claims (train, dev):
        {
          "claim-X": {
            "claim_text": "...",
            "claim_label": "SUPPORTS" | "REFUTES" | "NOT_ENOUGH_INFO" | "DISPUTED",
            "evidences": ["evidence-Y", ...]
          }
        }

    Unlabeled claims (test):
        {
          "claim-X": { "claim_text": "..." }
        }

    Evidence corpus:
        { "evidence-Y": "passage text", ... }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from src.data.schema import Claim, ClaimLabel, Evidence

PathLike = Union[str, Path]


def load_evidence(path: PathLike) -> dict[str, Evidence]:
    """Load the evidence corpus into a dict keyed by evidence id."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict at root of {path}, got {type(raw).__name__}")

    out: dict[str, Evidence] = {}
    for ev_id, text in raw.items():
        if not isinstance(text, str):
            raise ValueError(
                f"Evidence value for {ev_id!r} must be str, "
                f"got {type(text).__name__}"
            )
        out[ev_id] = Evidence(id=ev_id, text=text)
    return out


def load_claims(path: PathLike) -> dict[str, Claim]:
    """Load claims into a dict keyed by claim id.

    Auto-detects labeled vs unlabeled format on a per-claim basis.
    Raises ValueError on malformed input (missing claim_text, invalid label,
    wrong types).
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict at root of {path}, got {type(raw).__name__}")

    out: dict[str, Claim] = {}
    for claim_id, fields in raw.items():
        out[claim_id] = _parse_claim(claim_id, fields)
    return out


def _parse_claim(claim_id: str, fields: object) -> Claim:
    if not isinstance(fields, dict):
        raise ValueError(
            f"Expected dict for {claim_id!r}, got {type(fields).__name__}"
        )

    if "claim_text" not in fields:
        raise ValueError(f"Missing 'claim_text' for {claim_id!r}")
    text = fields["claim_text"]

    label_str = fields.get("claim_label")
    if label_str is None:
        label: ClaimLabel | None = None
    else:
        try:
            label = ClaimLabel(label_str)
        except ValueError:
            raise ValueError(
                f"Invalid label {label_str!r} for {claim_id!r}. "
                f"Expected one of {ClaimLabel.values()}"
            )

    evidence_ids_raw = fields.get("evidences", [])
    if not isinstance(evidence_ids_raw, list):
        raise ValueError(
            f"'evidences' for {claim_id!r} must be list, "
            f"got {type(evidence_ids_raw).__name__}"
        )
    for ev_id in evidence_ids_raw:
        if not isinstance(ev_id, str):
            raise ValueError(
                f"Evidence id in {claim_id!r} must be str, "
                f"got {type(ev_id).__name__}"
            )

    return Claim(
        id=claim_id,
        text=text,
        label=label,
        evidence_ids=tuple(evidence_ids_raw),
    )
