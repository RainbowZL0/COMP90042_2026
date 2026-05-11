"""Tests for hard-negative reranker data construction."""
from __future__ import annotations

from src.data.schema import Claim, ClaimLabel, Evidence
from src.retrieval.base import RetrievalResult, Retriever
from src.retrieval.rerank_data import (
    RerankExample,
    build_rerank_examples,
    load_rerank_examples,
    save_rerank_examples,
)


class FakeRetriever(Retriever):
    def __init__(self, mapping: dict[str, list[str]]):
        self.mapping = mapping

    def retrieve_batch(self, claim_texts: list[str], top_k: int):
        return [
            [RetrievalResult(evidence_id=eid, score=float(top_k - i)) for i, eid in
             enumerate(self.mapping[text][:top_k])]
            for text in claim_texts
        ]


def test_rerank_example_rejects_invalid_label():
    try:
        RerankExample("c", "e", "claim", "evidence", 2)
    except ValueError as e:
        assert "label" in str(e)
    else:
        raise AssertionError("Expected ValueError")


def test_build_examples_uses_gold_as_positive_and_excludes_gold_negatives():
    claims = {
        "claim-1": Claim(
            id="claim-1",
            text="CO2 causes warming",
            label=ClaimLabel.SUPPORTS,
            evidence_ids=("evidence-gold-1", "evidence-gold-2"),
        )
    }
    evidence = {
        "evidence-gold-1": Evidence("evidence-gold-1", "Gold warming evidence one"),
        "evidence-gold-2": Evidence("evidence-gold-2", "Gold warming evidence two"),
        "evidence-neg-1": Evidence("evidence-neg-1", "Hard negative one"),
        "evidence-neg-2": Evidence("evidence-neg-2", "Hard negative two"),
        "evidence-neg-3": Evidence("evidence-neg-3", "Hard negative three"),
        "evidence-neg-4": Evidence("evidence-neg-4", "Hard negative four"),
    }
    retriever = FakeRetriever(
        {
            "CO2 causes warming": [
                "evidence-gold-1",
                "evidence-neg-1",
                "evidence-neg-2",
                "evidence-gold-2",
                "evidence-neg-3",
                "evidence-neg-4",
            ]
        }
    )

    examples = build_rerank_examples(
        claims,
        evidence,
        retriever,
        candidate_k=6,
        negatives_per_positive=2,
        seed=7,
    )

    positives = [ex for ex in examples if ex.label == 1]
    negatives = [ex for ex in examples if ex.label == 0]
    assert {ex.evidence_id for ex in positives} == {"evidence-gold-1", "evidence-gold-2"}
    assert len(negatives) == 4
    assert not ({ex.evidence_id for ex in negatives} & {"evidence-gold-1", "evidence-gold-2"})


def test_examples_jsonl_roundtrip(tmp_path):
    examples = [
        RerankExample("c1", "e1", "claim", "evidence", 1),
        RerankExample("c1", "e2", "claim", "negative", 0),
    ]
    path = tmp_path / "pairs.jsonl"
    save_rerank_examples(examples, path)
    restored = load_rerank_examples(path)
    assert restored == examples
