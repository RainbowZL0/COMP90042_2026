"""Project-wide configuration.

Single source of truth for paths, seeds, and (later) hyperparameters.
Later phases will extend `Config` rather than scattering hyperparams.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Project root = parent of src/ (this file lives at src/config.py)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class Paths:
    """All file paths used by the project."""

    data: Path = DATA_DIR
    outputs: Path = OUTPUTS_DIR

    train_claims: Path = DATA_DIR / "train-claims.json"
    dev_claims: Path = DATA_DIR / "dev-claims.json"
    test_claims: Path = DATA_DIR / "test-claims-unlabelled.json"
    dev_baseline: Path = DATA_DIR / "dev-claims-baseline.json"
    evidence: Path = DATA_DIR / "evidence.json"


@dataclass(frozen=True)
class Config:
    """Top-level config. Extend in later phases (retrieval, classification)."""

    seed: int = 42
    paths: Paths = field(default_factory=Paths)


CONFIG = Config()


def get_device() -> str:
    """Return the best available device name for PyTorch.

    Returns 'cuda' if CUDA available, 'mps' on Apple Silicon, else 'cpu'.
    Returns 'cpu' if torch is not installed (so this can be imported
    in non-ML contexts like tests).
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
