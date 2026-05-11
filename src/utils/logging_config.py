"""Centralized logging setup.

Module name is `logging_config` (not `logging`) to avoid shadowing the
standard library.
"""
from __future__ import annotations

import logging
import sys

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"
_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger. Idempotent — safe to call repeatedly."""
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=_FMT, datefmt=_DATEFMT))

    root = logging.getLogger()
    root.setLevel(level)
    # Replace any prior handlers so logs don't duplicate when this is
    # re-imported from a notebook or test runner.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the project's standard configuration."""
    configure_logging()
    return logging.getLogger(name)
