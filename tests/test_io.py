"""Tests for src/utils/io.py.

The most important test here is `TestEvalPyCompatibility`: it runs the real
eval.py against our output to confirm format compatibility. If this test
ever fails, every prediction we ship is broken — fix immediately.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.data.schema import ClaimLabel, Prediction
from src.utils.io import read_predictions, write_predictions


@pytest.fixture
def sample_predictions() -> dict[str, Prediction]:
    return {
        "claim-1": Prediction(
            claim_id="claim-1",
            claim_text="some claim",
            claim_label=ClaimLabel.SUPPORTS,
            evidence_ids=("evidence-1", "evidence-2"),
        ),
        "claim-2": Prediction(
            claim_id="claim-2",
            claim_text="another claim",
            claim_label=ClaimLabel.DISPUTED,
            evidence_ids=("evidence-3",),
        ),
    }


class TestRoundTrip:
    def test_write_then_read_preserves_data(
        self, sample_predictions: dict[str, Prediction], tmp_path: Path
    ):
        out = tmp_path / "preds.json"
        write_predictions(sample_predictions, out)
        loaded = read_predictions(out)

        assert set(loaded.keys()) == set(sample_predictions.keys())
        for cid, pred in sample_predictions.items():
            assert loaded[cid].claim_label == pred.claim_label
            assert loaded[cid].evidence_ids == pred.evidence_ids
            assert loaded[cid].claim_text == pred.claim_text

    def test_creates_parent_directories(
        self, sample_predictions: dict[str, Prediction], tmp_path: Path
    ):
        out = tmp_path / "deep" / "nested" / "preds.json"
        write_predictions(sample_predictions, out)
        assert out.exists()

    def test_inconsistent_claim_id_raises(self, tmp_path: Path):
        bad = {
            "claim-1": Prediction(
                claim_id="claim-DIFFERENT",
                claim_text="x",
                claim_label=ClaimLabel.SUPPORTS,
                evidence_ids=("evidence-1",),
            )
        }
        with pytest.raises(ValueError, match="Inconsistent claim id"):
            write_predictions(bad, tmp_path / "preds.json")


class TestOutputFormat:
    """Validate the on-disk JSON shape matches eval.py's expectations."""

    def test_required_fields_present(
        self, sample_predictions: dict[str, Prediction], tmp_path: Path
    ):
        out = tmp_path / "preds.json"
        write_predictions(sample_predictions, out)
        with out.open() as f:
            raw = json.load(f)

        for cid in sample_predictions:
            assert "claim_label" in raw[cid]
            assert "evidences" in raw[cid]
            assert isinstance(raw[cid]["evidences"], list)
            assert raw[cid]["claim_label"] in ClaimLabel.values()

    def test_evidence_list_not_set_or_tuple(
        self, sample_predictions: dict[str, Prediction], tmp_path: Path
    ):
        # eval.py does `type(predictions[cid]["evidences"]) == list` (strict).
        # JSON only round-trips lists, but verify explicitly.
        out = tmp_path / "preds.json"
        write_predictions(sample_predictions, out)
        with out.open() as f:
            raw = json.load(f)
        for cid in sample_predictions:
            assert type(raw[cid]["evidences"]) is list


class TestEvalPyCompatibility:
    """End-to-end: write our format, run the real eval.py, parse results.

    This test is the final guardrail against format drift. It needs eval.py
    in a known location; if it's unavailable, skip rather than fail.
    """

    @staticmethod
    def _find_eval_py() -> Path | None:
        # Try known locations in order: project root sibling, then /mnt/project
        candidates = [
            Path(__file__).resolve().parent.parent / "eval.py",
            Path("/mnt/project/eval.py"),
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def test_perfect_predictions_score_one(
        self, sample_predictions: dict[str, Prediction], tmp_path: Path
    ):
        eval_py = self._find_eval_py()
        if eval_py is None:
            pytest.skip("eval.py not found in known locations")

        # Construct a ground truth that exactly matches predictions
        # (so a working eval.py should report F=1, A=1, hmean=1).
        gt: dict[str, dict] = {}
        for cid, pred in sample_predictions.items():
            gt[cid] = {
                "claim_text": pred.claim_text,
                "claim_label": pred.claim_label.value,
                "evidences": list(pred.evidence_ids),
            }
        gt_path = tmp_path / "gt.json"
        gt_path.write_text(json.dumps(gt), encoding="utf-8")

        pred_path = tmp_path / "pred.json"
        write_predictions(sample_predictions, pred_path)

        # Copy eval.py into tmp_path so we don't pollute the real project.
        local_eval = tmp_path / "eval.py"
        shutil.copy(eval_py, local_eval)

        result = subprocess.run(
            [
                sys.executable,
                str(local_eval),
                "--predictions",
                str(pred_path),
                "--groundtruth",
                str(gt_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, (
            f"eval.py exited non-zero. stderr:\n{result.stderr}\n"
            f"stdout:\n{result.stdout}"
        )
        # Perfect match → both metrics 1.0
        assert "F-score (F)    = 1.0" in result.stdout
        assert "Accuracy (A) = 1.0" in result.stdout
        assert "Harmonic Mean of F and A          = 1.0" in result.stdout
