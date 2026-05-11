"""Tests for seed and logging utilities."""
from __future__ import annotations

import logging
import random

import pytest

from src.utils.logging_config import configure_logging, get_logger
from src.utils.seed import set_seed


class TestSetSeed:
    def test_python_random_is_deterministic(self):
        set_seed(42)
        a = [random.random() for _ in range(5)]
        set_seed(42)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_different_seeds_produce_different_streams(self):
        set_seed(1)
        a = random.random()
        set_seed(2)
        b = random.random()
        assert a != b

    def test_non_int_raises(self):
        with pytest.raises(TypeError):
            set_seed("42")  # type: ignore[arg-type]


class TestLogging:
    def test_get_logger_returns_logger(self):
        log = get_logger("test_module")
        assert isinstance(log, logging.Logger)

    def test_configure_logging_is_idempotent(self):
        configure_logging()
        n_handlers_first = len(logging.getLogger().handlers)
        configure_logging()
        n_handlers_second = len(logging.getLogger().handlers)
        assert n_handlers_first == n_handlers_second
