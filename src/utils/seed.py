"""Global seed control for reproducibility."""
from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    """Seed Python's random, NumPy, and PyTorch (if installed).

    Also sets PYTHONHASHSEED for hash-based determinism in fresh subprocesses.
    Note: this does not enable PyTorch's deterministic algorithms — that is a
    larger tradeoff (slower, restricts ops) and is left to caller code that
    actually trains models.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
