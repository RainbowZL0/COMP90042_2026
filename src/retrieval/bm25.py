"""BM25 retrieval over the evidence corpus.

Uses bm25s (https://github.com/xhluca/bm25s) for ~100x speedup over the
pure-Python rank_bm25, plus built-in disk persistence.

Typical usage:
    # First time (slow):
    retriever = BM25Retriever.from_evidence(evidence_dict)
    retriever.save(path)

    # Subsequent runs (fast):
    retriever = BM25Retriever.load(path)
    results = retriever.retrieve("some claim", top_k=100)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import bm25s

from src.data.schema import Evidence
from src.retrieval.base import RetrievalResult, Retriever

PathLike = Union[str, Path]

_META_FILENAME = "meta.json"


class BM25Retriever(Retriever):
    """BM25 over a fixed evidence corpus.

    Public construction:
      - `BM25Retriever.from_evidence(...)` to build from scratch
      - `BM25Retriever.load(...)`           to restore from disk

    Direct `__init__` is for internal use only.
    """

    def __init__(
        self,
        bm25_index: bm25s.BM25,
        evidence_ids: tuple[str, ...],
        stopwords: Optional[str] = "en",
    ):
        self._index = bm25_index
        self._evidence_ids = evidence_ids
        self._stopwords = stopwords

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_evidence(
        cls,
        evidence: dict[str, Evidence],
        stopwords: Optional[str] = "en",
    ) -> "BM25Retriever":
        """Build a fresh BM25 index from an evidence dict.

        Evidence ids are stored in sorted order so the (int_index <-> id)
        mapping is deterministic across runs.
        """
        if not evidence:
            raise ValueError("Cannot build BM25 index over empty evidence")

        evidence_ids = tuple(sorted(evidence.keys()))
        corpus = [evidence[eid].text for eid in evidence_ids]

        corpus_tokens = bm25s.tokenize(
            corpus, stopwords=stopwords, show_progress=False
        )
        index = bm25s.BM25()
        index.index(corpus_tokens, show_progress=False)

        return cls(bm25_index=index, evidence_ids=evidence_ids, stopwords=stopwords)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_batch(
        self, claim_texts: list[str], top_k: int
    ) -> list[list[RetrievalResult]]:
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        if not claim_texts:
            return []

        # bm25s clamps k internally if it exceeds corpus size, but make it
        # explicit for clarity.
        effective_k = min(top_k, len(self._evidence_ids))

        query_tokens = bm25s.tokenize(
            claim_texts, stopwords=self._stopwords, show_progress=False
        )
        # (n_queries, k) int64 indices and float32 scores
        indices, scores = self._index.retrieve(
            query_tokens, k=effective_k, show_progress=False
        )

        out: list[list[RetrievalResult]] = []
        for q_indices, q_scores in zip(indices, scores):
            q_results = [
                RetrievalResult(
                    evidence_id=self._evidence_ids[int(idx)],
                    score=float(score),
                )
                for idx, score in zip(q_indices, q_scores)
            ]
            out.append(q_results)
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: PathLike) -> None:
        """Persist the index to a directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # bm25s saves its own files (params, indptr, data, vocab, indices)
        self._index.save(str(path))

        # We separately save the id-mapping and stopword config
        meta = {
            "evidence_ids": list(self._evidence_ids),
            "stopwords": self._stopwords,
        }
        (path / _META_FILENAME).write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, path: PathLike) -> "BM25Retriever":
        path = Path(path)
        meta_path = path / _META_FILENAME
        if not meta_path.exists():
            raise FileNotFoundError(
                f"BM25 meta file missing at {meta_path}. "
                f"Was the index built with BM25Retriever.save()?"
            )

        index = bm25s.BM25.load(str(path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(
            bm25_index=index,
            evidence_ids=tuple(meta["evidence_ids"]),
            stopwords=meta.get("stopwords", "en"),
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._evidence_ids)
