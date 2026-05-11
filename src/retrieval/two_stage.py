"""Two-stage retrieval: BM25 candidate generation + cross-encoder reranking."""
from __future__ import annotations

from typing import Protocol, Sequence

from src.data.schema import Evidence
from src.retrieval.base import RetrievalResult, Retriever


class SupportsRerankScoring(Protocol):
    """Minimal protocol implemented by CrossEncoderReranker and test fakes."""

    def predict_scores(
        self,
        claim_texts: Sequence[str],
        evidences: Sequence[Evidence],
        batch_size: int = 32,
    ) -> list[float]: ...


class TwoStageRetriever(Retriever):
    """First retrieve many candidates, then rerank with a cross-encoder."""

    def __init__(
        self,
        first_stage: Retriever,
        reranker: SupportsRerankScoring,
        evidence_corpus: dict[str, Evidence],
        candidate_k: int = 500,
        rerank_batch_size: int = 32,
    ):
        if candidate_k <= 0:
            raise ValueError("candidate_k must be > 0")
        if rerank_batch_size <= 0:
            raise ValueError("rerank_batch_size must be > 0")
        if not evidence_corpus:
            raise ValueError("evidence_corpus must be non-empty")
        self.first_stage = first_stage
        self.reranker = reranker
        self.evidence_corpus = evidence_corpus
        self.candidate_k = candidate_k
        self.rerank_batch_size = rerank_batch_size

    def retrieve_batch(
        self, claim_texts: list[str], top_k: int
    ) -> list[list[RetrievalResult]]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        scored = self.score_candidates_batch(claim_texts)
        return [items[:top_k] for items in scored]

    def score_candidates_batch(
        self, claim_texts: list[str]
    ) -> list[list[RetrievalResult]]:
        """Return all candidate_k results reranked by cross-encoder score."""
        if not claim_texts:
            return []

        first_stage_results = self.first_stage.retrieve_batch(
            claim_texts, self.candidate_k
        )
        out: list[list[RetrievalResult]] = []

        for claim_text, candidates in zip(claim_texts, first_stage_results):
            # Deduplicate while preserving first-stage rank. This matters when a
            # future hybrid retriever returns union candidates.
            seen: set[str] = set()
            candidate_ids: list[str] = []
            candidate_evidences: list[Evidence] = []
            for result in candidates:
                ev_id = result.evidence_id
                if ev_id in seen:
                    continue
                ev = self.evidence_corpus.get(ev_id)
                if ev is None:
                    continue
                seen.add(ev_id)
                candidate_ids.append(ev_id)
                candidate_evidences.append(ev)

            if not candidate_evidences:
                out.append([])
                continue

            repeated_claims = [claim_text] * len(candidate_evidences)
            scores = self.reranker.predict_scores(
                repeated_claims,
                candidate_evidences,
                batch_size=self.rerank_batch_size,
            )
            reranked = [
                RetrievalResult(evidence_id=ev_id, score=float(score))
                for ev_id, score in zip(candidate_ids, scores)
            ]
            reranked.sort(key=lambda r: r.score, reverse=True)
            out.append(reranked)
        return out

    def retrieve_batch_threshold(
        self,
        claim_texts: list[str],
        threshold: float,
        min_k: int = 1,
        max_k: int = 5,
    ) -> list[list[RetrievalResult]]:
        """Dynamic-K retrieval using a reranker-score threshold.

        Always returns at least min_k and at most max_k results per claim.
        """
        if min_k <= 0:
            raise ValueError("min_k must be > 0")
        if max_k < min_k:
            raise ValueError("max_k must be >= min_k")

        scored = self.score_candidates_batch(claim_texts)
        out: list[list[RetrievalResult]] = []
        for items in scored:
            selected = [r for r in items[:max_k] if r.score >= threshold]
            if len(selected) < min_k:
                selected = items[:min_k]
            out.append(selected[:max_k])
        return out
