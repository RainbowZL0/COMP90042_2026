"""Evaluate retrieval-only quality: Recall@N over a range of N values.

This is the core scientific output of Phase 2. It answers:

  "How many BM25 candidates does the Phase 3 reranker need to see?"

We want to pick N where Recall@N is near its asymptote (so the reranker
has access to most ground-truth evidence) but not so large that reranker
inference becomes slow. Typical choice: smallest N where Recall@N > 0.85.

The output also gives per-claim percentiles, so we see whether failures
are concentrated in a few hard claims or spread across the dev set.

Usage:
    python scripts/eval_retrieval.py --split dev
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import CONFIG  # noqa: E402
from src.data.loader import load_claims  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.utils.logging_config import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

log = get_logger("eval_retrieval")

N_VALUES = [5, 10, 20, 50, 100, 200, 500, 1000]


def recall_at_n(retrieved_ids: list[str], gt_ids: set[str]) -> float:
    if not gt_ids:
        return 0.0
    return len(set(retrieved_ids) & gt_ids) / len(gt_ids)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(p * (len(s) - 1))
    return s[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=["dev", "train"], default="dev",
        help="Which split to evaluate on (default: dev)",
    )
    args = parser.parse_args()

    set_seed(CONFIG.seed)

    index_path = CONFIG.paths.outputs / "bm25_index"
    if not index_path.exists():
        log.error(
            f"Index not found at {index_path}. "
            f"Run `python scripts/build_index.py` first."
        )
        sys.exit(1)
    log.info(f"Loading BM25 index from {index_path}")
    retriever = BM25Retriever.load(index_path)
    log.info(f"Index loaded ({len(retriever):,} passages)")

    split_path = {
        "dev": CONFIG.paths.dev_claims,
        "train": CONFIG.paths.train_claims,
    }[args.split]
    log.info(f"Loading claims from {split_path}")
    claims_dict = load_claims(split_path)
    labeled = [c for c in claims_dict.values() if c.is_labeled and c.evidence_ids]
    log.info(f"Evaluating on {len(labeled)} labeled claims with ground-truth evidence")

    max_n = max(N_VALUES)
    log.info(f"Retrieving top_{max_n} per claim (this may take a while)...")
    t0 = time.time()
    claim_texts = [c.text for c in labeled]
    results = retriever.retrieve_batch(claim_texts, max_n)
    elapsed = time.time() - t0
    log.info(
        f"Done in {elapsed:.1f}s ({elapsed / len(labeled) * 1000:.1f}ms/claim)"
    )

    # ---- Compute Recall@N per claim, then aggregate ----
    log.info("=" * 70)
    log.info(f"{'N':>5}  {'Recall@N (mean)':>16}  {'P50':>8}  {'P10':>8}  {'@1.0':>8}")
    log.info("-" * 70)
    for n in N_VALUES:
        per_claim_recalls = []
        n_perfect = 0
        for claim, claim_results in zip(labeled, results):
            top_n_ids = [r.evidence_id for r in claim_results[:n]]
            r = recall_at_n(top_n_ids, set(claim.evidence_ids))
            per_claim_recalls.append(r)
            if r >= 1.0 - 1e-9:
                n_perfect += 1
        mean_r = sum(per_claim_recalls) / len(per_claim_recalls)
        p50 = percentile(per_claim_recalls, 0.50)
        p10 = percentile(per_claim_recalls, 0.10)
        pct_perfect = n_perfect / len(per_claim_recalls) * 100
        log.info(
            f"{n:>5}  {mean_r:>16.4f}  {p50:>8.3f}  {p10:>8.3f}  "
            f"{pct_perfect:>7.1f}%"
        )
    log.info("=" * 70)
    log.info("Reading this table:")
    log.info("  Recall@N (mean) = average per-claim recall at cut-off N")
    log.info("  P50             = median (half of claims achieve at least this)")
    log.info("  P10             = 10th percentile (worst-served claims)")
    log.info("  @1.0            = fraction of claims with ALL GT in top-N (perfect)")
    log.info("")
    log.info("Phase 3 design implication:")
    log.info("  Choose reranker input N where mean Recall@N nears its asymptote.")
    log.info("  Common choice: smallest N with Recall@N >= 0.85 (or 0.90).")


if __name__ == "__main__":
    main()
