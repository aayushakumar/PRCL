"""Unit tests for SimCLR transforms."""

import torch
from PIL import Image

from prcl.ssl.methods.simclr_transforms import get_simclr_transform


class TestSimCLRTransforms:
    def test_train_transform_returns_two_views(self):
        transform = get_simclr_transform("cifar10", train=True)
        img = Image.fromarray((torch.rand(32, 32, 3).numpy() * 255).astype("uint8"))
        v1, v2 = transform(img)
        assert v1.shape == (3, 32, 32)
        assert v2.shape == (3, 32, 32)

    def test_eval_transform_returns_single(self):
        transform = get_simclr_transform("cifar10", train=False)
        img = Image.fromarray((torch.rand(32, 32, 3).numpy() * 255).astype("uint8"))
        result = transform(img)
        assert result.shape == (3, 32, 32)

    def test_stl10_sizing(self):
        transform = get_simclr_transform("stl10", train=True)
        img = Image.fromarray((torch.rand(96, 96, 3).numpy() * 255).astype("uint8"))
        v1, v2 = transform(img)
        assert v1.shape == (3, 96, 96)

    def test_two_views_are_different(self):
        transform = get_simclr_transform("cifar10", train=True)
        img = Image.fromarray((torch.rand(32, 32, 3).numpy() * 255).astype("uint8"))
        v1, v2 = transform(img)
        # Views should almost certainly be different due to random augmentation
        assert not torch.equal(v1, v2)
