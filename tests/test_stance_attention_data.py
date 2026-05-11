from src.classification.stance_attention_data import (
    STANCE_NEUTRAL,
    STANCE_REFUTE,
    STANCE_SUPPORT,
    example_from_evidences,
    examples_from_evidence_batches,
)
from src.data.schema import Claim, ClaimLabel, Evidence


def test_supports_gold_evidence_gets_support_stance_label():
    claim = Claim("c1", "claim", ClaimLabel.SUPPORTS, ("e1", "e2"))
    evidences = [Evidence("e1", "a"), Evidence("e2", "b")]
    ex = example_from_evidences(claim, evidences, max_evidences=3, gold_evidence_ids={"e1", "e2"})
    assert ex is not None
    assert ex.stance_labels == (STANCE_SUPPORT, STANCE_SUPPORT, STANCE_NEUTRAL)
    assert ex.stance_mask == (1, 1, 0)
    assert ex.evidence_mask == (1, 1, 0)


def test_refutes_gold_evidence_gets_refute_stance_label():
    claim = Claim("c1", "claim", ClaimLabel.REFUTES, ("e1",))
    ex = example_from_evidences(claim, [Evidence("e1", "a")], max_evidences=2, gold_evidence_ids={"e1"})
    assert ex is not None
    assert ex.stance_labels == (STANCE_REFUTE, STANCE_NEUTRAL)
    assert ex.stance_mask == (1, 0)


def test_disputed_no_direct_stance_supervision():
    claim = Claim("c1", "claim", ClaimLabel.DISPUTED, ("e1", "e2"))
    ex = example_from_evidences(
        claim, [Evidence("e1", "a"), Evidence("e2", "b")], max_evidences=2, gold_evidence_ids={"e1", "e2"}
    )
    assert ex is not None
    assert ex.stance_mask == (0, 0)


def test_retrieved_non_gold_support_evidence_not_stance_supervised():
    claim = Claim("c1", "claim", ClaimLabel.SUPPORTS, ("gold",))
    evidences = [Evidence("wrong", "a")]
    ex = examples_from_evidence_batches([claim], [evidences], max_evidences=1)[0]
    assert ex.stance_labels == (STANCE_NEUTRAL,)
    assert ex.stance_mask == (0,)
