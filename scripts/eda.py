"""Exploratory data analysis.

Prints key statistics that drive design decisions in later phases:

  - Class distribution (drives loss weighting, class imbalance handling)
  - Ground-truth evidence count per claim (drives top-K selection in retrieval)
  - Claim and evidence text length distributions (drives max-seq-len budget
    for cross-encoder inputs)

Run from project root:
    python scripts/eda.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# Make `src` importable when run as a plain script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import CONFIG  # noqa: E402
from src.data.loader import load_claims, load_evidence  # noqa: E402
from src.data.schema import ClaimLabel  # noqa: E402
from src.utils.logging_config import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

log = get_logger("eda")


def percentile(values: list[int], p: float) -> float:
    """Linear-interpolation percentile (matches numpy.percentile default)."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return s[f] * (c - k) + s[c] * (k - f)


def describe(values: list[int], unit: str) -> str:
    if not values:
        return f"  (empty, unit={unit})"
    return (
        f"  min={min(values)}, max={max(values)}, "
        f"mean={sum(values) / len(values):.1f}, "
        f"p50={percentile(values, 0.50):.0f}, "
        f"p90={percentile(values, 0.90):.0f}, "
        f"p99={percentile(values, 0.99):.0f}  ({unit})"
    )


def report_claims(name: str, claims_path: Path) -> None:
    if not claims_path.exists():
        log.warning(f"{name}: file not found at {claims_path}, skipping")
        return

    claims = load_claims(claims_path)
    log.info(f"=== {name}: {len(claims)} claims ===")

    is_labeled_set = any(c.is_labeled for c in claims.values())

    if is_labeled_set:
        labels = Counter(c.label.value for c in claims.values() if c.is_labeled)
        total = sum(labels.values())
        log.info("Label distribution:")
        for lab in ClaimLabel.values():
            n = labels.get(lab, 0)
            pct = n / total * 100 if total else 0
            log.info(f"  {lab:18s}  {n:5d}  ({pct:5.1f}%)")

        ev_counts = [len(c.evidence_ids) for c in claims.values() if c.is_labeled]
        log.info("Ground-truth evidences per claim:")
        log.info(describe(ev_counts, "evidences"))
        # Full distribution helps choose the right top-K range
        dist = sorted(Counter(ev_counts).items())
        log.info(f"  exact counts: {dist}")

    log.info("Claim text length:")
    log.info(describe([len(c.text) for c in claims.values()], "chars"))
    log.info(describe([len(c.text.split()) for c in claims.values()], "words"))


def report_evidence(evidence_path: Path) -> None:
    if not evidence_path.exists():
        log.warning(f"evidence file not found at {evidence_path}, skipping")
        return

    log.info(f"=== Evidence corpus ===")
    log.info(f"Loading from {evidence_path} (this may take a moment for full corpus)...")
    ev = load_evidence(evidence_path)
    log.info(f"Total evidence passages: {len(ev):,}")

    char_lens = [len(e.text) for e in ev.values()]
    word_lens = [len(e.text.split()) for e in ev.values()]
    log.info("Evidence text length:")
    log.info(describe(char_lens, "chars"))
    log.info(describe(word_lens, "words"))


def main() -> None:
    set_seed(CONFIG.seed)

    log.info("=" * 70)
    log.info("EDA Report")
    log.info("=" * 70)

    report_claims("Train", CONFIG.paths.train_claims)
    report_claims("Dev", CONFIG.paths.dev_claims)
    report_claims("Test (unlabeled)", CONFIG.paths.test_claims)
    report_evidence(CONFIG.paths.evidence)

    log.info("=" * 70)
    log.info("Done. Use these stats for:")
    log.info("  - Class distribution -> loss weighting in classifier (Phase 4)")
    log.info("  - Ground-truth evidence counts -> top-K range to sweep (Phase 3)")
    log.info("  - Text length p99 -> max_seq_length for cross-encoder (Phase 3)")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
