"""Shared test fixtures."""

import sys
from pathlib import Path

import pytest
import torch

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_batch():
    """Small random batch for shape tests."""
    return torch.randn(8, 3, 32, 32)
