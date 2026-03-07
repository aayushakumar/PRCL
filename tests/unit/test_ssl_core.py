"""Unit tests for core SSL components."""

import pytest
import torch

from prcl.ssl.backbones.resnet import get_backbone
from prcl.ssl.heads.projection import ProjectionHead
from prcl.ssl.losses.infonce import InfoNCELoss
from prcl.ssl.train_loop import SimCLRModel


class TestBackbone:
    def test_resnet18_output_shape(self, small_batch):
        backbone, feat_dim = get_backbone("resnet18")
        assert feat_dim == 512
        h = backbone(small_batch)
        assert h.shape == (8, 512)

    def test_resnet50_output_shape(self):
        backbone, feat_dim = get_backbone("resnet50")
        assert feat_dim == 2048
        x = torch.randn(4, 3, 32, 32)
        h = backbone(x)
        assert h.shape == (4, 2048)

    def test_unknown_backbone_raises(self):
        with pytest.raises(ValueError, match="Unknown backbone"):
            get_backbone("vgg16")


class TestProjectionHead:
    def test_output_shape(self):
        proj = ProjectionHead(512, 2048, 128)
        x = torch.randn(4, 512)
        z = proj(x)
        assert z.shape == (4, 128)

    def test_custom_dims(self):
        proj = ProjectionHead(256, 1024, 64)
        x = torch.randn(4, 256)
        z = proj(x)
        assert z.shape == (4, 64)


class TestInfoNCELoss:
    def test_loss_is_scalar(self):
        criterion = InfoNCELoss(temperature=0.5)
        z = torch.randn(8, 128)  # 4 pairs
        loss = criterion(z)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_loss_gradient_flows(self):
        criterion = InfoNCELoss(temperature=0.5)
        z = torch.randn(8, 128, requires_grad=True)
        loss = criterion(z)
        loss.backward()
        assert z.grad is not None
        assert z.grad.shape == (8, 128)

    def test_perfect_pairs_lower_loss(self):
        """Identical positive pairs should have lower loss than random."""
        criterion = InfoNCELoss(temperature=0.5)
        # Random
        z_random = torch.randn(8, 128)
        loss_random = criterion(z_random)
        # Make positive pairs very similar
        z_aligned = torch.randn(4, 128)
        z_aligned = torch.cat([z_aligned, z_aligned + 0.01 * torch.randn(4, 128)], dim=0)
        # Interleave: [z0, z0', z1, z1', ...]
        z_interleaved = torch.zeros(8, 128)
        z_interleaved[0::2] = z_aligned[:4]
        z_interleaved[1::2] = z_aligned[4:]
        loss_aligned = criterion(z_interleaved)
        assert loss_aligned.item() < loss_random.item()

    def test_different_batch_sizes(self):
        criterion = InfoNCELoss(temperature=0.5)
        for n in [2, 4, 16, 32]:
            z = torch.randn(2 * n, 128)
            loss = criterion(z)
            assert loss.ndim == 0
            assert not torch.isnan(loss)


class TestSimCLRModel:
    def test_forward(self, small_batch):
        model = SimCLRModel("resnet18", projection_dim=128, projection_hidden_dim=2048)
        h, z = model(small_batch)
        assert h.shape == (8, 512)
        assert z.shape == (8, 128)

    def test_encode(self, small_batch):
        model = SimCLRModel("resnet18")
        h = model.encode(small_batch)
        assert h.shape == (8, 512)
