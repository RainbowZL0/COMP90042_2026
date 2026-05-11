"""Tests for src/data/schema.py."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.data.schema import Claim, ClaimLabel, Evidence, Prediction


class TestClaimLabel:
    def test_all_four_labels_exist(self):
        assert ClaimLabel("SUPPORTS") == ClaimLabel.SUPPORTS
        assert ClaimLabel("REFUTES") == ClaimLabel.REFUTES
        assert ClaimLabel("NOT_ENOUGH_INFO") == ClaimLabel.NOT_ENOUGH_INFO
        assert ClaimLabel("DISPUTED") == ClaimLabel.DISPUTED

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError):
            ClaimLabel("MAYBE")

    def test_values_returns_all_four(self):
        vals = ClaimLabel.values()
        assert set(vals) == {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"}
        assert len(vals) == 4

    def test_string_compatibility(self):
        # str-Enum: equal to the underlying string for JSON serialization.
        assert ClaimLabel.SUPPORTS == "SUPPORTS"
        assert ClaimLabel.SUPPORTS.value == "SUPPORTS"


class TestEvidence:
    def test_construct_valid(self):
        ev = Evidence(id="evidence-1", text="some text")
        assert ev.id == "evidence-1"
        assert ev.text == "some text"

    def test_empty_text_is_allowed(self):
        # Some entries in the corpus may genuinely be short or empty;
        # we don't gate on text length, only on type.
        Evidence(id="evidence-1", text="")

    def test_empty_id_raises(self):
        with pytest.raises(ValueError):
            Evidence(id="", text="x")

    def test_non_string_id_raises(self):
        with pytest.raises(ValueError):
            Evidence(id=123, text="x")  # type: ignore[arg-type]

    def test_non_string_text_raises(self):
        with pytest.raises(TypeError):
            Evidence(id="evidence-1", text=42)  # type: ignore[arg-type]

    def test_immutable(self):
        ev = Evidence(id="evidence-1", text="x")
        with pytest.raises(FrozenInstanceError):
            ev.id = "evidence-2"  # type: ignore[misc]


class TestClaim:
    def test_labeled_claim(self):
        c = Claim(
            id="claim-1",
            text="some claim",
            label=ClaimLabel.SUPPORTS,
            evidence_ids=("evidence-1", "evidence-2"),
        )
        assert c.is_labeled
        assert c.label == ClaimLabel.SUPPORTS
        assert c.evidence_ids == ("evidence-1", "evidence-2")

    def test_unlabeled_claim_defaults(self):
        c = Claim(id="claim-1", text="some claim")
        assert not c.is_labeled
        assert c.label is None
        assert c.evidence_ids == ()

    def test_label_must_be_enum_not_string(self):
        # Loader is responsible for converting str -> ClaimLabel; the schema
        # itself rejects raw strings to keep type guarantees strict.
        with pytest.raises(TypeError):
            Claim(id="claim-1", text="x", label="SUPPORTS")  # type: ignore[arg-type]

    def test_empty_text_raises(self):
        with pytest.raises(ValueError):
            Claim(id="claim-1", text="")

    def test_evidence_ids_must_be_tuple(self):
        with pytest.raises(TypeError):
            Claim(
                id="claim-1",
                text="x",
                evidence_ids=["evidence-1"],  # type: ignore[arg-type]
            )


class TestPrediction:
    def test_construct_valid(self):
        p = Prediction(
            claim_id="claim-1",
            claim_text="x",
            claim_label=ClaimLabel.SUPPORTS,
            evidence_ids=("evidence-1",),
        )
        assert p.claim_id == "claim-1"
        assert p.claim_label == ClaimLabel.SUPPORTS
        assert p.evidence_ids == ("evidence-1",)

    def test_empty_evidence_raises(self):
        # Spec: every prediction must retrieve at least 1 evidence.
        with pytest.raises(ValueError, match="at least one evidence"):
            Prediction(
                claim_id="claim-1",
                claim_text="x",
                claim_label=ClaimLabel.SUPPORTS,
                evidence_ids=(),
            )

    def test_label_must_be_enum(self):
        with pytest.raises(TypeError):
            Prediction(
                claim_id="claim-1",
                claim_text="x",
                claim_label="SUPPORTS",  # type: ignore[arg-type]
                evidence_ids=("evidence-1",),
            )
