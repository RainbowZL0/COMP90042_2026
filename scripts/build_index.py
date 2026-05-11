"""Build the BM25 index over the evidence corpus and save to disk.

Run once after evidence.json is in place. Subsequent scripts load the
cached index in ~1-2 seconds rather than rebuilding (which takes ~60-90s
for the full 1.2M corpus).

Usage:
    python scripts/build_index.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import CONFIG  # noqa: E402
from src.data.loader import load_evidence  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.utils.logging_config import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

log = get_logger("build_index")


def main() -> None:
    set_seed(CONFIG.seed)

    log.info(f"Loading evidence from {CONFIG.paths.evidence}")
    t0 = time.time()
    evidence = load_evidence(CONFIG.paths.evidence)
    log.info(f"Loaded {len(evidence):,} evidences in {time.time() - t0:.1f}s")

    log.info("Building BM25 index (typically 60-90s for 1.2M passages)...")
    t0 = time.time()
    retriever = BM25Retriever.from_evidence(evidence)
    log.info(f"Index built in {time.time() - t0:.1f}s")

    index_path = CONFIG.paths.outputs / "bm25_index"
    log.info(f"Saving index to {index_path}")
    retriever.save(index_path)
    log.info("Done.")


if __name__ == "__main__":
    main()
