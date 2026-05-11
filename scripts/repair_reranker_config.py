"""Repair reranker checkpoints saved with invalid single-logit HF config.

Run once if loading outputs/reranker/model fails with:
    problem_type="single_label_classification" requires num_labels > 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import CONFIG
from src.retrieval.reranker import repair_single_logit_config
from src.utils.logging_config import configure_logging, get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

LOGGER = get_logger("repair_reranker_config")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=CONFIG.paths.outputs / "reranker" / "model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    config_path = args.model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    changed = repair_single_logit_config(args.model_dir)
    if changed:
        LOGGER.info("Repaired %s", config_path)
    else:
        LOGGER.info("No repair needed for %s", config_path)


if __name__ == "__main__":
    main()
