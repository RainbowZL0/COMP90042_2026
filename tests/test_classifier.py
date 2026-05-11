"""Tests for src/classification/majority.py."""
from __future__ import annotations

import pytest

from src.classification.base import Classifier
from src.classification.majority import MajorityClassifier
from src.data.schema import Claim, ClaimLabel, Evidence


@pytest.fixture
def evidences() -> list[Evidence]:
    return [Evidence(id=f"evidence-{i}", text=f"text {i}") for i in range(3)]


class TestMajorityClassifier:
    def test_implements_classifier_interface(self):
        c = MajorityClassifier(ClaimLabel.SUPPORTS)
        assert isinstance(c, Classifier)

    def test_invalid_label_type_raises(self):
        with pytest.raises(TypeError):
            MajorityClassifier(label="SUPPORTS")  # type: ignore[arg-type]

    def test_always_returns_configured_label(self, evidences):
        c = MajorityClassifier(ClaimLabel.DISPUTED)
        for claim_text in ["anything", "something else", ""]:
            # Empty text is allowed by the classifier; only Claim itself rejects it
            assert c.classify("nonempty", evidences) == ClaimLabel.DISPUTED

    def test_classify_batch_length_matches_input(self, evidences):
        c = MajorityClassifier(ClaimLabel.SUPPORTS)
        out = c.classify_batch(["a", "b", "c"], [evidences, evidences, evidences])
        assert out == [ClaimLabel.SUPPORTS] * 3

    def test_mismatched_batch_sizes_raise(self, evidences):
        c = MajorityClassifier(ClaimLabel.SUPPORTS)
        with pytest.raises(ValueError, match="same length"):
            c.classify_batch(["a", "b"], [evidences])


class TestFit:
    def _make_claims(
        self, label_counts: dict[ClaimLabel, int]
    ) -> dict[str, Claim]:
        claims = {}
        i = 0
        for label, n in label_counts.items():
            for _ in range(n):
                cid = f"claim-{i}"
                claims[cid] = Claim(
                    id=cid,
                    text="t",
                    label=label,
                    evidence_ids=("evidence-1",),
                )
                i += 1
        return claims

    def test_fit_picks_most_frequent_label(self):
        claims = self._make_claims(
            {
                ClaimLabel.SUPPORTS: 5,
                ClaimLabel.REFUTES: 2,
                ClaimLabel.DISPUTED: 1,
            }
        )
        c = MajorityClassifier.fit(claims)
        assert c.label == ClaimLabel.SUPPORTS

    def test_fit_works_on_real_train_distribution(self):
        # Mirrors the dev's actual label frequencies (~44/27/18/12 %)
        claims = self._make_claims(
            {
                ClaimLabel.SUPPORTS: 519,
                ClaimLabel.NOT_ENOUGH_INFO: 386,
                ClaimLabel.REFUTES: 199,
                ClaimLabel.DISPUTED: 124,
            }
        )
        c = MajorityClassifier.fit(claims)
        assert c.label == ClaimLabel.SUPPORTS

    def test_fit_unlabeled_data_raises(self):
        unlabeled = {
            "claim-1": Claim(id="claim-1", text="t"),
        }
        with pytest.raises(ValueError, match="unlabeled"):
            MajorityClassifier.fit(unlabeled)

    def test_fit_empty_dict_raises(self):
        with pytest.raises(ValueError):
            MajorityClassifier.fit({})
