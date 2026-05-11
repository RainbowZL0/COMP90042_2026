"""End-to-end claim verification pipeline.

Two stages: Stage 1 retrieves top_k evidences via a Retriever, Stage 2
labels the claim via a Classifier. The Pipeline owns no model weights
itself — it only orchestrates.
"""
from __future__ import annotations

from src.classification.base import Classifier
from src.data.schema import Claim, Evidence, Prediction
from src.retrieval.base import Retriever


class Pipeline:
    """Two-stage fact-checking pipeline (retrieve → classify)."""

    def __init__(
        self,
        retriever: Retriever,
        classifier: Classifier,
        evidence_corpus: dict[str, Evidence],
        top_k: int,
    ):
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        if not evidence_corpus:
            raise ValueError("evidence_corpus must be non-empty")
        self._retriever = retriever
        self._classifier = classifier
        self._corpus = evidence_corpus
        self._top_k = top_k

    @property
    def top_k(self) -> int:
        return self._top_k

    def predict(self, claim: Claim) -> Prediction:
        return self.predict_batch([claim])[claim.id]

    def predict_batch(self, claims: list[Claim]) -> dict[str, Prediction]:
        if not claims:
            return {}

        claim_texts = [c.text for c in claims]

        # ---- Stage 1: retrieval ----
        retrieved = self._retriever.retrieve_batch(claim_texts, self._top_k)

        # Map evidence_ids to Evidence objects. Defensively drop ids
        # not in the corpus (shouldn't happen, but better than crashing
        # on a single bad mapping).
        evidences_per_claim: list[list[Evidence]] = []
        retrieved_ids_per_claim: list[tuple[str, ...]] = []
        for results in retrieved:
            evs: list[Evidence] = []
            ids: list[str] = []
            for r in results:
                ev = self._corpus.get(r.evidence_id)
                if ev is not None:
                    evs.append(ev)
                    ids.append(r.evidence_id)
            evidences_per_claim.append(evs)
            retrieved_ids_per_claim.append(tuple(ids))

        # ---- Stage 2: classification ----
        labels = self._classifier.classify_batch(claim_texts, evidences_per_claim)

        # ---- Assemble Predictions ----
        out: dict[str, Prediction] = {}
        for claim, label, ev_ids in zip(claims, labels, retrieved_ids_per_claim):
            if not ev_ids:
                # Spec: every prediction must include >=1 evidence.
                # If the retriever returned nothing valid (which would mean
                # an empty corpus or all ids missing), there's a bug
                # upstream; fail loudly rather than silently mislabel.
                raise RuntimeError(
                    f"Pipeline: no valid evidence retrieved for {claim.id}. "
                    f"Check that retriever and corpus are consistent."
                )
            out[claim.id] = Prediction(
                claim_id=claim.id,
                claim_text=claim.text,
                claim_label=label,
                evidence_ids=ev_ids,
            )
        return out
