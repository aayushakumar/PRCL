"""Unit tests for PRCL defense components — probes, PCF, robust loss, quarantine."""

import pytest
import torch
import torch.nn as nn

from prcl.defenses.prcl.pcf import NeighborOverlapScorer, ProbeAlignmentScorer
from prcl.defenses.prcl.probes import (
    BlurProbe,
    DesaturationProbe,
    FreqLowpassProbe,
    OcclusionProbe,
    get_probe_transforms,
)
from prcl.defenses.prcl.quarantine import QuarantineManager
from prcl.defenses.prcl.robust_loss import WeightedInfoNCELoss
from prcl.defenses.prcl.thresholds import BatchMinMaxNormalizer, RollingZScoreNormalizer

# ---------- Probe transforms ----------


class TestProbeTransforms:
    def test_blur_preserves_shape(self, small_batch):
        probe = BlurProbe()
        result = probe(small_batch)
        assert result.shape == small_batch.shape

    def test_blur_single_image(self):
        img = torch.randn(3, 32, 32)
        probe = BlurProbe()
        result = probe(img)
        assert result.shape == (3, 32, 32)

    def test_occlusion_preserves_shape(self, small_batch):
        probe = OcclusionProbe()
        result = probe(small_batch)
        assert result.shape == small_batch.shape

    def test_occlusion_zeros_some_pixels(self):
        img = torch.ones(3, 32, 32)
        probe = OcclusionProbe(occlusion_ratio=0.25)
        result = probe(img)
        assert (result == 0).any()  # some pixels should be zero

    def test_freq_lowpass_preserves_shape(self, small_batch):
        probe = FreqLowpassProbe()
        result = probe(small_batch)
        assert result.shape == small_batch.shape

    def test_desaturation_preserves_shape(self, small_batch):
        probe = DesaturationProbe()
        result = probe(small_batch)
        assert result.shape == small_batch.shape

    def test_desaturation_channels_equal(self):
        img = torch.randn(3, 32, 32)
        probe = DesaturationProbe()
        result = probe(img)
        # All 3 channels should be identical after desaturation
        assert torch.allclose(result[0], result[1])
        assert torch.allclose(result[1], result[2])

    def test_registry_lookup(self):
        probes = get_probe_transforms(["blur", "occlusion"])
        assert len(probes) == 2
        assert isinstance(probes[0], BlurProbe)
        assert isinstance(probes[1], OcclusionProbe)

    def test_unknown_probe_raises(self):
        with pytest.raises(ValueError, match="Unknown probe"):
            get_probe_transforms(["nonexistent_probe"])


# ---------- Score normalization ----------


class TestNormalizers:
    def test_rolling_zscore_output_range(self):
        norm = RollingZScoreNormalizer()
        scores = torch.randn(32)
        result = norm.normalize(scores)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_rolling_zscore_updates_stats(self):
        norm = RollingZScoreNormalizer()
        assert norm.running_mean is None
        norm.normalize(torch.randn(32))
        assert norm.running_mean is not None

    def test_rolling_zscore_reset(self):
        norm = RollingZScoreNormalizer()
        norm.normalize(torch.randn(32))
        norm.reset()
        assert norm.running_mean is None

    def test_batch_minmax_output_range(self):
        norm = BatchMinMaxNormalizer()
        scores = torch.randn(32)
        result = norm.normalize(scores)
        assert torch.isclose(result.min(), torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(result.max(), torch.tensor(1.0), atol=1e-6)

    def test_batch_minmax_constant_input(self):
        norm = BatchMinMaxNormalizer()
        scores = torch.ones(32) * 5.0
        result = norm.normalize(scores)
        assert torch.all(result == 0)  # degenerate case


# ---------- PCF scorers ----------


class TestProbeAlignmentScorer:
    def _make_encoder(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 64),
        )

    def test_score_shape_and_range(self):
        scorer = ProbeAlignmentScorer(probe_types=["blur"], normalize="batch_minmax")
        encoder = self._make_encoder()
        images = torch.randn(8, 3, 32, 32)
        scores = scorer.compute_scores(images, encoder)
        assert scores.shape == (8,)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_scores_detached(self):
        scorer = ProbeAlignmentScorer(probe_types=["blur"], normalize="none")
        encoder = self._make_encoder()
        images = torch.randn(8, 3, 32, 32)
        scores = scorer.compute_scores(images, encoder)
        assert not scores.requires_grad

    def test_multiple_probes(self):
        scorer = ProbeAlignmentScorer(
            probe_types=["blur", "occlusion", "desaturation"],
            normalize="rolling_zscore",
        )
        encoder = self._make_encoder()
        images = torch.randn(8, 3, 32, 32)
        scores = scorer.compute_scores(images, encoder)
        assert scores.shape == (8,)

    def test_with_precomputed_representations(self):
        scorer = ProbeAlignmentScorer(probe_types=["blur"], normalize="none")
        encoder = self._make_encoder()
        images = torch.randn(8, 3, 32, 32)
        with torch.no_grad():
            clean_h = encoder(images)
        scores = scorer.compute_scores(images, encoder, clean_representations=clean_h)
        assert scores.shape == (8,)


class TestNeighborOverlapScorer:
    def _make_encoder(self):
        return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 64))

    def test_score_shape_and_range(self):
        scorer = NeighborOverlapScorer(
            probe_types=["blur"], neighbor_k=3, normalize="batch_minmax"
        )
        encoder = self._make_encoder()
        images = torch.randn(8, 3, 32, 32)
        scores = scorer.compute_scores(images, encoder)
        assert scores.shape == (8,)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_small_batch_clamps_k(self):
        scorer = NeighborOverlapScorer(
            probe_types=["blur"], neighbor_k=100, normalize="none"
        )
        encoder = self._make_encoder()
        images = torch.randn(4, 3, 32, 32)
        scores = scorer.compute_scores(images, encoder)
        assert scores.shape == (4,)


# ---------- Weighted InfoNCE ----------


class TestWeightedInfoNCELoss:
    def test_standard_mode(self):
        loss_fn = WeightedInfoNCELoss(temperature=0.5, mode="none")
        z = torch.randn(8, 128)
        loss, stats = loss_fn(z)
        assert loss.ndim == 0
        assert loss.item() > 0
        assert stats["mean_pos_weight"] == 1.0

    def test_soft_weight_mode(self):
        loss_fn = WeightedInfoNCELoss(temperature=0.5, mode="soft_weight")
        z = torch.randn(8, 128)
        q = torch.tensor([0.1, 0.9, 0.3, 0.7])  # 4 samples
        loss, stats = loss_fn(z, suspicion_scores=q)
        assert loss.ndim == 0
        assert stats["mean_pos_weight"] < 1.0
        assert "max_suspicion_in_batch" in stats

    def test_high_suspicion_lowers_weight(self):
        loss_fn = WeightedInfoNCELoss(
            temperature=0.5, mode="soft_weight", lambda_pos=1.0, w_min=0.0
        )
        z = torch.randn(8, 128)

        # All clean (low suspicion)
        q_clean = torch.zeros(4)
        _, stats_clean = loss_fn(z, q_clean)

        # All suspicious
        q_sus = torch.ones(4)
        _, stats_sus = loss_fn(z, q_sus)

        assert stats_clean["mean_pos_weight"] > stats_sus["mean_pos_weight"]

    def test_grad_cap_mode(self):
        loss_fn = WeightedInfoNCELoss(
            temperature=0.5, mode="soft_weight+grad_cap",
            grad_cap_enabled=True, grad_cap_value=2.0,
        )
        z = torch.randn(8, 128)
        q = torch.rand(4)
        loss, stats = loss_fn(z, q)
        assert loss.ndim == 0

    def test_trim_mode(self):
        loss_fn = WeightedInfoNCELoss(
            temperature=0.5, mode="trim", trim_alpha=0.5, lambda_pos=0.0,
        )
        z = torch.randn(8, 128)
        q = torch.tensor([0.1, 0.9, 0.3, 0.7])
        loss, stats = loss_fn(z, q)
        assert loss.ndim == 0

    def test_gradient_flows(self):
        loss_fn = WeightedInfoNCELoss(temperature=0.5, mode="soft_weight")
        z = torch.randn(8, 128, requires_grad=True)
        q = torch.rand(4)
        loss, _ = loss_fn(z, q)
        loss.backward()
        assert z.grad is not None

    def test_none_scores_fallback(self):
        loss_fn = WeightedInfoNCELoss(temperature=0.5, mode="soft_weight")
        z = torch.randn(8, 128)
        loss, stats = loss_fn(z, suspicion_scores=None)
        assert stats["mean_pos_weight"] == 1.0


# ---------- Quarantine ----------


class TestQuarantineManager:
    def test_initialization(self):
        qm = QuarantineManager(dataset_size=100)
        assert qm.quarantined.sum() == 0
        assert len(qm.ema_scores) == 100

    def test_score_update(self):
        qm = QuarantineManager(dataset_size=100, threshold=0.8)
        indices = torch.tensor([0, 1, 2])
        scores = torch.tensor([0.5, 0.9, 0.3])
        qm.update_scores(indices, scores)
        assert qm.update_counts[0] == 1
        assert qm.ema_scores[1] == pytest.approx(0.9, abs=0.01)

    def test_quarantine_update(self):
        qm = QuarantineManager(dataset_size=10, threshold=0.8, percentile=1.0)
        indices = torch.arange(10)
        scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.95])
        qm.update_scores(indices, scores)
        qm.update_quarantine()
        # Samples 8 and 9 should be quarantined (>0.8)
        assert qm.quarantined[9]
        assert qm.quarantined[8]
        assert not qm.quarantined[0]

    def test_is_quarantined(self):
        qm = QuarantineManager(dataset_size=10, threshold=0.8, percentile=1.0)
        indices = torch.arange(10)
        scores = torch.linspace(0, 1, 10)
        qm.update_scores(indices, scores)
        qm.update_quarantine()
        result = qm.is_quarantined(torch.tensor([0, 9]))
        assert not result[0]
        assert result[1]

    def test_stats(self):
        qm = QuarantineManager(dataset_size=10, threshold=0.9, percentile=1.0)
        stats = qm.get_stats()
        assert "quarantine_count" in stats
        assert "quarantine_ratio" in stats

    def test_epoch_reevaluation(self):
        qm = QuarantineManager(dataset_size=10, threshold=0.5, reevaluate_every=2, percentile=1.0)
        indices = torch.arange(10)
        scores = torch.linspace(0, 1, 10)
        qm.update_scores(indices, scores)
        # Should trigger quarantine update at epoch 1 (every 2)
        qm.on_epoch_end(1)
        assert qm.quarantined.sum() > 0
