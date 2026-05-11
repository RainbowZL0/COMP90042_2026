"""Tests for src/retrieval/bm25.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.schema import Evidence
from src.retrieval.base import RetrievalResult, Retriever
from src.retrieval.bm25 import BM25Retriever


@pytest.fixture
def tiny_evidence() -> dict[str, Evidence]:
    """Small corpus with topically distinct documents."""
    return {
        "evidence-cat": Evidence(
            id="evidence-cat",
            text="The cat sat on the mat in the warm afternoon sun.",
        ),
        "evidence-dog": Evidence(
            id="evidence-dog",
            text="The dog chased a tennis ball across the green park.",
        ),
        "evidence-climate": Evidence(
            id="evidence-climate",
            text="Climate sensitivity measures how much warming results from CO2 doubling.",
        ),
        "evidence-ocean": Evidence(
            id="evidence-ocean",
            text="Sea levels rose by several inches over the last century.",
        ),
    }


@pytest.fixture
def bm25(tiny_evidence: dict[str, Evidence]) -> BM25Retriever:
    return BM25Retriever.from_evidence(tiny_evidence)


class TestConstruction:
    def test_from_evidence_returns_retriever(self, bm25: BM25Retriever):
        assert isinstance(bm25, Retriever)
        assert isinstance(bm25, BM25Retriever)
        assert len(bm25) == 4

    def test_empty_evidence_raises(self):
        with pytest.raises(ValueError, match="empty"):
            BM25Retriever.from_evidence({})


class TestRetrieve:
    def test_single_query_returns_list_of_results(self, bm25: BM25Retriever):
        results = bm25.retrieve("cat", top_k=2)
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_top_relevant_doc_ranked_first(self, bm25: BM25Retriever):
        # Query mentions "climate" and "CO2" — should pull the climate doc to rank 1
        results = bm25.retrieve("CO2 climate sensitivity", top_k=4)
        assert results[0].evidence_id == "evidence-climate"

    def test_scores_are_descending(self, bm25: BM25Retriever):
        results = bm25.retrieve("cat dog warm sea climate", top_k=4)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_clamped_to_corpus_size(self, bm25: BM25Retriever):
        results = bm25.retrieve("anything", top_k=1000)
        # bm25s returns at most len(corpus) results
        assert len(results) <= len(bm25)

    def test_invalid_top_k_raises(self, bm25: BM25Retriever):
        with pytest.raises(ValueError, match="top_k"):
            bm25.retrieve("cat", top_k=0)
        with pytest.raises(ValueError, match="top_k"):
            bm25.retrieve("cat", top_k=-1)


class TestRetrieveBatch:
    def test_batch_preserves_order(self, bm25: BM25Retriever):
        queries = ["cat warm", "ocean sea level", "tennis ball"]
        batch_results = bm25.retrieve_batch(queries, top_k=1)
        assert len(batch_results) == 3
        # Each query should retrieve the topically aligned doc first
        assert batch_results[0][0].evidence_id == "evidence-cat"
        assert batch_results[1][0].evidence_id == "evidence-ocean"
        assert batch_results[2][0].evidence_id == "evidence-dog"

    def test_empty_batch_returns_empty(self, bm25: BM25Retriever):
        assert bm25.retrieve_batch([], top_k=5) == []

    def test_batch_matches_single_retrieve(self, bm25: BM25Retriever):
        """Single retrieve should produce identical results to batch-of-1."""
        single = bm25.retrieve("climate", top_k=3)
        batch = bm25.retrieve_batch(["climate"], top_k=3)[0]
        assert [r.evidence_id for r in single] == [r.evidence_id for r in batch]


class TestPersistence:
    def test_save_load_roundtrip_preserves_retrieval(
        self, bm25: BM25Retriever, tmp_path: Path
    ):
        save_dir = tmp_path / "index"
        bm25.save(save_dir)
        loaded = BM25Retriever.load(save_dir)

        assert len(loaded) == len(bm25)

        original = bm25.retrieve("climate CO2", top_k=4)
        restored = loaded.retrieve("climate CO2", top_k=4)
        # Same ids and scores in the same order
        assert [r.evidence_id for r in original] == [r.evidence_id for r in restored]
        for a, b in zip(original, restored):
            assert abs(a.score - b.score) < 1e-6

    def test_load_missing_meta_raises(self, tmp_path: Path):
        # bm25s.BM25.load may itself raise; we just need a clean error
        # rather than an obscure one.
        with pytest.raises((FileNotFoundError, Exception)):
            BM25Retriever.load(tmp_path / "does_not_exist")
