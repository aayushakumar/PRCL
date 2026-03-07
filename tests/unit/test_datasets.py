"""Unit tests for dataset wrappers."""

from unittest.mock import MagicMock

import numpy as np
import torch
from torch.utils.data import Dataset

from prcl.datasets.cifar import PoisonAwareDataset


class FakeDataset(Dataset):
    """Minimal dataset for testing."""
    def __init__(self, size=100):
        self.data = [torch.randn(3, 32, 32) for _ in range(size)]
        self.targets = [i % 10 for i in range(size)]  # cycling labels 0-9

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class TestPoisonAwareDataset:
    def test_clean_dataset(self):
        base = FakeDataset(100)
        ds = PoisonAwareDataset(base)
        assert len(ds) == 100
        assert ds.num_poisoned == 0
        img, idx = ds[0]
        assert isinstance(idx, int)

    def test_with_poison_indices(self):
        base = FakeDataset(100)
        poison_idx = np.array([0, 5, 10, 15])
        ds = PoisonAwareDataset(base, poison_indices=poison_idx)
        assert ds.num_poisoned == 4
        assert ds.poison_mask[0]
        assert not ds.poison_mask[1]
        assert ds.poison_mask[5]

    def test_poison_fn_applied(self):
        base = FakeDataset(100)
        poison_idx = np.array([0])
        marker = torch.ones(3, 32, 32) * 999
        ds = PoisonAwareDataset(
            base, poison_indices=poison_idx,
            poison_fn=lambda img: marker,
        )
        img, idx = ds[0]
        assert torch.equal(img, marker)

    def test_labels_accessible(self):
        base = FakeDataset(100)
        ds = PoisonAwareDataset(base)
        labels = ds.labels
        assert len(labels) == 100

    def test_transform_applied(self):
        base = FakeDataset(10)
        transform = MagicMock(return_value=(torch.randn(3, 32, 32), torch.randn(3, 32, 32)))
        ds = PoisonAwareDataset(base, transform=transform)
        result, idx = ds[0]
        assert transform.called
