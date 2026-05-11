"""Shared test fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# Minimal samples that exercise both labeled and unlabeled formats and
# all four claim labels.
SAMPLE_TRAIN: dict = {
    "claim-1937": {
        "claim_text": "Higher CO2 helps plants grow.",
        "claim_label": "DISPUTED",
        "evidences": ["evidence-1", "evidence-2"],
    },
    "claim-126": {
        "claim_text": "El Nino drove record temperatures.",
        "claim_label": "REFUTES",
        "evidences": ["evidence-3"],
    },
    "claim-2510": {
        "claim_text": "PDO switched to a cool phase in 1946.",
        "claim_label": "SUPPORTS",
        "evidences": ["evidence-1"],
    },
    "claim-851": {
        "claim_text": "Four degrees warmer the oceans were higher.",
        "claim_label": "NOT_ENOUGH_INFO",
        "evidences": ["evidence-2", "evidence-3"],
    },
}

SAMPLE_TEST: dict = {
    "claim-2967": {"claim_text": "Waste heat contributes 0.028 W/m2."},
    "claim-979": {"claim_text": "Warm weather worsened the drought."},
}

SAMPLE_EVIDENCE: dict = {
    "evidence-1": "First evidence passage about CO2 concentrations.",
    "evidence-2": "Second evidence passage on plant biology.",
    "evidence-3": "Third evidence passage on Pacific Decadal Oscillation.",
}


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def train_claims_file(tmp_path: Path) -> Path:
    return _write_json(tmp_path / "train.json", SAMPLE_TRAIN)


@pytest.fixture
def test_claims_file(tmp_path: Path) -> Path:
    return _write_json(tmp_path / "test.json", SAMPLE_TEST)


@pytest.fixture
def evidence_file(tmp_path: Path) -> Path:
    return _write_json(tmp_path / "evidence.json", SAMPLE_EVIDENCE)
