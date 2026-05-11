"""Run the full pipeline (BM25 retrieval + Majority classifier) and write
predictions in eval.py's format.

This is Phase 2's primary deliverable: the first concrete F/A/hmean numbers
that all later phases compare against.

Usage:
    python scripts/run_pipeline.py --split dev --top_k 4
    python eval.py --predictions outputs/dev-predictions-k4.json \\
                   --groundtruth data/dev-claims.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.classification.majority import MajorityClassifier  # noqa: E402
from src.config import CONFIG  # noqa: E402
from src.data.loader import load_claims, load_evidence  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.utils.io import write_predictions  # noqa: E402
from src.utils.logging_config import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

log = get_logger("pipeline")

SPLITS = {
    "train": CONFIG.paths.train_claims,
    "dev": CONFIG.paths.dev_claims,
    "test": CONFIG.paths.test_claims,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=list(SPLITS), default="dev",
        help="Which claim file to predict on (default: dev)",
    )
    parser.add_argument(
        "--top_k", type=int, default=4,
        help="How many evidence to retrieve per claim (default: 4)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path (default: outputs/<split>-predictions-k<K>.json)",
    )
    args = parser.parse_args()

    set_seed(CONFIG.seed)

    # ---- Load evidence corpus and BM25 index ----
    log.info(f"Loading evidence corpus from {CONFIG.paths.evidence}")
    t0 = time.time()
    evidence = load_evidence(CONFIG.paths.evidence)
    log.info(f"Loaded {len(evidence):,} evidences in {time.time() - t0:.1f}s")

    index_path = CONFIG.paths.outputs / "bm25_index"
    if not index_path.exists():
        log.error(
            f"BM25 index not found at {index_path}. "
            f"Run `python scripts/build_index.py` first."
        )
        sys.exit(1)
    log.info(f"Loading BM25 index from {index_path}")
    retriever = BM25Retriever.load(index_path)
    log.info(f"Index loaded ({len(retriever):,} passages)")

    # ---- Fit majority classifier on training data ----
    log.info(f"Fitting majority classifier on {CONFIG.paths.train_claims}")
    train_claims = load_claims(CONFIG.paths.train_claims)
    classifier = MajorityClassifier.fit(train_claims)
    log.info(f"Majority label = {classifier.label.value}")

    pipeline = Pipeline(
        retriever=retriever,
        classifier=classifier,
        evidence_corpus=evidence,
        top_k=args.top_k,
    )

    # ---- Run on the requested split ----
    claims_path = SPLITS[args.split]
    log.info(f"Loading claims from {claims_path}")
    claims_dict = load_claims(claims_path)
    claims_list = list(claims_dict.values())
    log.info(f"{len(claims_list)} claims to predict")

    log.info(f"Running pipeline (top_k={args.top_k})...")
    t0 = time.time()
    predictions = pipeline.predict_batch(claims_list)
    elapsed = time.time() - t0
    log.info(
        f"Pipeline completed in {elapsed:.1f}s "
        f"({elapsed / len(claims_list) * 1000:.1f}ms/claim)"
    )

    out_path = args.output or (
        CONFIG.paths.outputs / f"{args.split}-predictions-k{args.top_k}.json"
    )
    write_predictions(predictions, out_path)
    log.info(f"Predictions written to {out_path}")

    # ---- Print eval command for convenience ----
    if args.split == "dev":
        log.info("=" * 70)
        log.info("To evaluate, run:")
        log.info(
            f"  python eval.py --predictions {out_path} "
            f"--groundtruth {CONFIG.paths.dev_claims}"
        )
        log.info("=" * 70)


if __name__ == "__main__":
    main()
