from src.classification.joint_data import (
    ID_TO_LABEL,
    LABELS,
    LABEL_TO_ID,
    build_evidence_bundle,
    compute_balanced_class_weights,
    examples_from_gold_evidence,
)
from src.data.schema import Claim, ClaimLabel, Evidence


def test_label_maps_are_roundtrip():
    assert len(LABELS) == 4
    for label in LABELS:
        assert ID_TO_LABEL[LABEL_TO_ID[label]] == label


def test_build_evidence_bundle_respects_max_evidences():
    evs = [Evidence(id=f"e{i}", text=f"text {i}") for i in range(5)]
    bundle = build_evidence_bundle(evs, max_evidences=3)
    assert "text 0" in bundle
    assert "text 2" in bundle
    assert "text 3" not in bundle


def test_examples_from_gold_evidence_skips_unlabeled_and_missing_evidence():
    claims = [
        Claim(
            id="c1",
            text="claim one",
            label=ClaimLabel.SUPPORTS,
            evidence_ids=("e1", "missing"),
        ),
        Claim(id="c2", text="claim two", label=None, evidence_ids=("e1",)),
    ]
    evidence = {"e1": Evidence(id="e1", text="evidence one")}
    examples = examples_from_gold_evidence(claims, evidence, max_evidences=3)
    assert len(examples) == 1
    assert examples[0].claim_id == "c1"
    assert examples[0].label == ClaimLabel.SUPPORTS
    assert examples[0].evidence_text == "evidence one"


def test_compute_balanced_class_weights():
    claims = [
        Claim(id="c1", text="a", label=ClaimLabel.SUPPORTS, evidence_ids=("e",)),
        Claim(id="c2", text="b", label=ClaimLabel.SUPPORTS, evidence_ids=("e",)),
        Claim(id="c3", text="c", label=ClaimLabel.REFUTES, evidence_ids=("e",)),
        Claim(id="c4", text="d", label=ClaimLabel.NOT_ENOUGH_INFO, evidence_ids=("e",)),
        Claim(id="c5", text="e", label=ClaimLabel.DISPUTED, evidence_ids=("e",)),
    ]
    evidence = {"e": Evidence(id="e", text="evidence")}
    examples = examples_from_gold_evidence(claims, evidence)
    weights = compute_balanced_class_weights(examples)
    assert len(weights) == 4
    assert weights[LABEL_TO_ID[ClaimLabel.SUPPORTS]] < weights[LABEL_TO_ID[ClaimLabel.REFUTES]]
