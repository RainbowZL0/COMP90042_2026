"""Tests for src/pipeline.py — end-to-end Stage1+Stage2 orchestration."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.classification.majority import MajorityClassifier
from src.data.schema import Claim, ClaimLabel, Evidence
from src.pipeline import Pipeline
from src.retrieval.base import RetrievalResult, Retriever
from src.utils.io import write_predictions


class FakeRetriever(Retriever):
    """A retriever that returns hard-coded results — for isolating Pipeline logic."""

    def __init__(self, results_by_text: dict[str, list[RetrievalResult]]):
        self._results = results_by_text

    def retrieve_batch(
        self, claim_texts: list[str], top_k: int
    ) -> list[list[RetrievalResult]]:
        out = []
        for text in claim_texts:
            results = self._results.get(text, [])
            out.append(list(results[:top_k]))
        return out


@pytest.fixture
def evidence_corpus() -> dict[str, Evidence]:
    return {
        f"evidence-{i}": Evidence(id=f"evidence-{i}", text=f"text {i}")
        for i in range(10)
    }


@pytest.fixture
def claims() -> list[Claim]:
    return [
        Claim(id="claim-1", text="first claim"),
        Claim(id="claim-2", text="second claim"),
    ]


@pytest.fixture
def fake_retriever() -> FakeRetriever:
    return FakeRetriever(
        {
            "first claim": [
                RetrievalResult("evidence-1", 0.9),
                RetrievalResult("evidence-2", 0.5),
                RetrievalResult("evidence-3", 0.1),
            ],
            "second claim": [
                RetrievalResult("evidence-5", 0.8),
                RetrievalResult("evidence-6", 0.3),
            ],
        }
    )


class TestPipelineConstruction:
    def test_invalid_top_k_raises(self, fake_retriever, evidence_corpus):
        c = MajorityClassifier(ClaimLabel.SUPPORTS)
        with pytest.raises(ValueError, match="top_k"):
            Pipeline(fake_retriever, c, evidence_corpus, top_k=0)

    def test_empty_corpus_raises(self, fake_retriever):
        c = MajorityClassifier(ClaimLabel.SUPPORTS)
        with pytest.raises(ValueError, match="non-empty"):
            Pipeline(fake_retriever, c, {}, top_k=4)


class TestPredict:
    def test_predict_single_claim(self, fake_retriever, evidence_corpus, claims):
        pipeline = Pipeline(
            retriever=fake_retriever,
            classifier=MajorityClassifier(ClaimLabel.SUPPORTS),
            evidence_corpus=evidence_corpus,
            top_k=2,
        )
        pred = pipeline.predict(claims[0])
        assert pred.claim_id == "claim-1"
        assert pred.claim_label == ClaimLabel.SUPPORTS
        assert pred.evidence_ids == ("evidence-1", "evidence-2")

    def test_predict_batch_respects_top_k(
        self, fake_retriever, evidence_corpus, claims
    ):
        pipeline = Pipeline(
            retriever=fake_retriever,
            classifier=MajorityClassifier(ClaimLabel.SUPPORTS),
            evidence_corpus=evidence_corpus,
            top_k=2,
        )
        preds = pipeline.predict_batch(claims)
        assert len(preds) == 2
        for pred in preds.values():
            assert len(pred.evidence_ids) <= 2

    def test_unknown_evidence_id_is_dropped_silently(
        self, evidence_corpus, claims
    ):
        # Retriever returns an id that's NOT in the corpus
        retriever = FakeRetriever(
            {
                "first claim": [
                    RetrievalResult("evidence-99999", 1.0),  # not in corpus
                    RetrievalResult("evidence-1", 0.5),
                ],
            }
        )
        pipeline = Pipeline(
            retriever=retriever,
            classifier=MajorityClassifier(ClaimLabel.SUPPORTS),
            evidence_corpus=evidence_corpus,
            top_k=2,
        )
        pred = pipeline.predict(claims[0])
        # Unknown id dropped, real id retained
        assert pred.evidence_ids == ("evidence-1",)

    def test_all_unknown_ids_raises_loudly(self, evidence_corpus, claims):
        # Pipeline must enforce the "at least one evidence" spec; if all
        # retrieved ids are bogus, that's a bug we want to surface, not hide.
        retriever = FakeRetriever(
            {
                "first claim": [RetrievalResult("nonexistent", 1.0)],
            }
        )
        pipeline = Pipeline(
            retriever=retriever,
            classifier=MajorityClassifier(ClaimLabel.SUPPORTS),
            evidence_corpus=evidence_corpus,
            top_k=2,
        )
        with pytest.raises(RuntimeError, match="no valid evidence"):
            pipeline.predict(claims[0])

    def test_empty_claim_list_returns_empty_dict(
        self, fake_retriever, evidence_corpus
    ):
        pipeline = Pipeline(
            retriever=fake_retriever,
            classifier=MajorityClassifier(ClaimLabel.SUPPORTS),
            evidence_corpus=evidence_corpus,
            top_k=4,
        )
        assert pipeline.predict_batch([]) == {}


class TestEndToEndWithEvalPy:
    """The single most important test in Phase 2.

    Builds a real (tiny) corpus, runs the real Pipeline (BM25 + Majority),
    writes predictions, runs the real eval.py, parses results.
    If this passes, the entire scaffolding works.
    """

    @staticmethod
    def _find_eval_py() -> Path | None:
        candidates = [
            Path(__file__).resolve().parent.parent / "eval.py",
            Path("/mnt/project/eval.py"),
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def test_full_pipeline_through_eval_py(self, tmp_path: Path):
        from src.retrieval.bm25 import BM25Retriever

        eval_py = self._find_eval_py()
        if eval_py is None:
            pytest.skip("eval.py not found")

        # Build a tiny corpus where a query has an obvious correct evidence
        corpus = {
            "evidence-climate-1": Evidence(
                id="evidence-climate-1",
                text="Climate sensitivity from CO2 doubling is 1.5 to 4.5 degrees.",
            ),
            "evidence-unrelated": Evidence(
                id="evidence-unrelated",
                text="The cat sat on the mat in the afternoon sun.",
            ),
            "evidence-ocean": Evidence(
                id="evidence-ocean",
                text="Sea levels have risen several inches in the last century.",
            ),
        }
        gt_claim = Claim(
            id="claim-1",
            text="CO2 doubling drives 1.5 to 4.5 degrees of warming",
            label=ClaimLabel.SUPPORTS,
            evidence_ids=("evidence-climate-1",),
        )

        retriever = BM25Retriever.from_evidence(corpus)
        classifier = MajorityClassifier(ClaimLabel.SUPPORTS)
        pipeline = Pipeline(
            retriever=retriever,
            classifier=classifier,
            evidence_corpus=corpus,
            top_k=1,
        )
        predictions = pipeline.predict_batch([gt_claim])

        # Write predictions and a matching ground truth file
        pred_path = tmp_path / "pred.json"
        write_predictions(predictions, pred_path)

        gt = {
            gt_claim.id: {
                "claim_text": gt_claim.text,
                "claim_label": gt_claim.label.value,
                "evidences": list(gt_claim.evidence_ids),
            }
        }
        gt_path = tmp_path / "gt.json"
        gt_path.write_text(json.dumps(gt), encoding="utf-8")

        # Copy eval.py somewhere local so we don't pollute the project tree
        local_eval = tmp_path / "eval.py"
        shutil.copy(eval_py, local_eval)

        result = subprocess.run(
            [
                sys.executable, str(local_eval),
                "--predictions", str(pred_path),
                "--groundtruth", str(gt_path),
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"eval.py failed: stderr={result.stderr}, stdout={result.stdout}"
        )
        # BM25 should retrieve the climate doc (matching the GT), majority
        # predicts SUPPORTS (matching the GT) → both metrics = 1.0
        assert "F-score (F)    = 1.0" in result.stdout
        assert "Accuracy (A) = 1.0" in result.stdout
