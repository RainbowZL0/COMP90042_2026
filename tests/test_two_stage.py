"""Tests for BM25 + reranker composition without loading neural models."""
from __future__ import annotations

from typing import Sequence

from src.data.schema import Evidence
from src.retrieval.base import RetrievalResult, Retriever
from src.retrieval.two_stage import TwoStageRetriever


class FakeFirstStage(Retriever):
    def retrieve_batch(self, claim_texts: list[str], top_k: int):
        return [
            [
                RetrievalResult("evidence-low", 10.0),
                RetrievalResult("evidence-high", 9.0),
                RetrievalResult("evidence-mid", 8.0),
            ][:top_k]
            for _ in claim_texts
        ]


class FakeReranker:
    def predict_scores(
        self,
        claim_texts: Sequence[str],
        evidences: Sequence[Evidence],
        batch_size: int = 32,
    ) -> list[float]:
        score_by_id = {
            "evidence-high": 3.0,
            "evidence-mid": 1.0,
            "evidence-low": -1.0,
        }
        return [score_by_id[e.id] for e in evidences]


def corpus() -> dict[str, Evidence]:
    return {
        "evidence-low": Evidence("evidence-low", "low relevance"),
        "evidence-mid": Evidence("evidence-mid", "medium relevance"),
        "evidence-high": Evidence("evidence-high", "high relevance"),
    }


def test_two_stage_reranks_by_cross_encoder_score():
    retriever = TwoStageRetriever(
        first_stage=FakeFirstStage(),
        reranker=FakeReranker(),
        evidence_corpus=corpus(),
        candidate_k=3,
    )
    results = retriever.retrieve("claim", top_k=2)
    assert [r.evidence_id for r in results] == ["evidence-high", "evidence-mid"]
    assert [r.score for r in results] == [3.0, 1.0]


def test_threshold_retrieval_enforces_min_and_max_k():
    retriever = TwoStageRetriever(
        first_stage=FakeFirstStage(),
        reranker=FakeReranker(),
        evidence_corpus=corpus(),
        candidate_k=3,
    )
    # threshold keeps only evidence-high.
    results = retriever.retrieve_batch_threshold(["claim"], threshold=2.0, min_k=1, max_k=3)[0]
    assert [r.evidence_id for r in results] == ["evidence-high"]

    # threshold too high would keep none, so min_k=1 falls back to best item.
    results = retriever.retrieve_batch_threshold(["claim"], threshold=10.0, min_k=1, max_k=3)[0]
    assert [r.evidence_id for r in results] == ["evidence-high"]

    # low threshold keeps all, then max_k truncates.
    results = retriever.retrieve_batch_threshold(["claim"], threshold=-10.0, min_k=1, max_k=2)[0]
    assert [r.evidence_id for r in results] == ["evidence-high", "evidence-mid"]
