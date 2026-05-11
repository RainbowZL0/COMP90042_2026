"""Tests for src/data/loader.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import CONFIG
from src.data.loader import load_claims, load_evidence
from src.data.schema import ClaimLabel


class TestLoadEvidence:
    def test_load_sample(self, evidence_file: Path):
        ev = load_evidence(evidence_file)
        assert len(ev) == 3
        assert ev["evidence-1"].text.startswith("First evidence")

    def test_each_evidence_has_correct_id_and_type(self, evidence_file: Path):
        ev = load_evidence(evidence_file)
        for ev_id, e in ev.items():
            assert e.id == ev_id
            assert isinstance(e.text, str)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_evidence(tmp_path / "does_not_exist.json")

    def test_non_dict_root_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
        with pytest.raises(ValueError, match="Expected dict"):
            load_evidence(bad)

    def test_non_string_value_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"evidence-1": 123}), encoding="utf-8")
        with pytest.raises(ValueError, match="must be str"):
            load_evidence(bad)


class TestLoadClaims:
    def test_load_labeled_train(self, train_claims_file: Path):
        claims = load_claims(train_claims_file)
        assert len(claims) == 4

        c = claims["claim-1937"]
        assert c.is_labeled
        assert c.label == ClaimLabel.DISPUTED
        assert c.evidence_ids == ("evidence-1", "evidence-2")

    def test_load_unlabeled_test(self, test_claims_file: Path):
        claims = load_claims(test_claims_file)
        assert len(claims) == 2

        c = claims["claim-2967"]
        assert not c.is_labeled
        assert c.label is None
        assert c.evidence_ids == ()

    def test_all_four_labels_parse(self, train_claims_file: Path):
        claims = load_claims(train_claims_file)
        labels_seen = {c.label for c in claims.values()}
        assert labels_seen == {
            ClaimLabel.SUPPORTS,
            ClaimLabel.REFUTES,
            ClaimLabel.NOT_ENOUGH_INFO,
            ClaimLabel.DISPUTED,
        }

    def test_invalid_label_raises_with_helpful_message(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text(
            json.dumps(
                {"claim-1": {"claim_text": "x", "claim_label": "MAYBE", "evidences": []}}
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Invalid label"):
            load_claims(bad)

    def test_missing_claim_text_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"claim-1": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="claim_text"):
            load_claims(bad)

    def test_evidences_must_be_list(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text(
            json.dumps(
                {
                    "claim-1": {
                        "claim_text": "x",
                        "claim_label": "SUPPORTS",
                        "evidences": "evidence-1",  # should be list
                    }
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="must be list"):
            load_claims(bad)

    def test_real_train_file_loads(self):
        """Smoke test against the user's actual train-claims.json.

        Skipped if the real file isn't yet in data/ (e.g. running on CI
        before data is provisioned). Uses CONFIG paths so it works on
        any OS without hard-coded paths.
        """
        if not CONFIG.paths.train_claims.exists():
            pytest.skip(
                f"real train file not present at {CONFIG.paths.train_claims}"
            )
        claims = load_claims(CONFIG.paths.train_claims)
        assert len(claims) > 0
        for c in claims.values():
            assert c.is_labeled
            assert isinstance(c.label, ClaimLabel)
