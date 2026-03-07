"""Quarantine buffer — optional exclusion of highly suspicious samples.

Tracks per-sample suspicion across epochs using exponential moving average.
Samples exceeding a threshold are quarantined (excluded from loss).
Quarantined samples are re-evaluated periodically and can be reinstated.

Default: DISABLED. This is an optional aggressive defense mode.
"""

import logging

import torch

logger = logging.getLogger(__name__)


class QuarantineManager:
    """Manages quarantine state for suspicious samples across epochs."""

    def __init__(
        self,
        dataset_size: int,
        threshold: float = 0.95,
        percentile: float = 0.99,
        reevaluate_every: int = 5,
        momentum: float = 0.3,
    ):
        self.dataset_size = dataset_size
        self.threshold = threshold
        self.percentile = percentile
        self.reevaluate_every = reevaluate_every
        self.momentum = momentum

        # EMA suspicion scores per sample
        self.ema_scores = torch.zeros(dataset_size)
        self.update_counts = torch.zeros(dataset_size, dtype=torch.long)
        self.quarantined = torch.zeros(dataset_size, dtype=torch.bool)
        self.epoch = 0

    def update_scores(self, indices: torch.Tensor, scores: torch.Tensor):
        """Update EMA suspicion scores for the given sample indices.

        Args:
            indices: Sample indices (shape varies, could be batch of original indices).
            scores: Corresponding suspicion scores q(x) ∈ [0,1].
        """
        indices = indices.cpu()
        scores = scores.cpu().detach()

        for idx, score in zip(indices, scores, strict=False):
            idx = idx.item()
            if self.update_counts[idx] == 0:
                self.ema_scores[idx] = score.item()
            else:
                self.ema_scores[idx] = (
                    (1 - self.momentum) * self.ema_scores[idx] + self.momentum * score.item()
                )
            self.update_counts[idx] += 1

    def update_quarantine(self):
        """Update quarantine set based on current EMA scores."""
        # Only consider samples that have been scored at least once
        scored_mask = self.update_counts > 0
        if scored_mask.sum() == 0:
            return

        scored_values = self.ema_scores[scored_mask]

        # Threshold-based quarantine
        threshold_mask = self.ema_scores > self.threshold

        # Percentile-based quarantine
        if self.percentile < 1.0:
            percentile_threshold = torch.quantile(scored_values, self.percentile)
            percentile_mask = self.ema_scores > percentile_threshold
            self.quarantined = (threshold_mask | percentile_mask) & scored_mask
        else:
            self.quarantined = threshold_mask & scored_mask

    def is_quarantined(self, indices: torch.Tensor) -> torch.Tensor:
        """Check if samples are quarantined.

        Returns a boolean mask of shape matching indices.
        """
        return self.quarantined[indices.cpu()].to(indices.device)

    def on_epoch_end(self, epoch: int):
        """Called at epoch end to optionally re-evaluate quarantine."""
        self.epoch = epoch
        if (epoch + 1) % self.reevaluate_every == 0:
            self.update_quarantine()
            n_quarantined = self.quarantined.sum().item()
            logger.info(
                f"Quarantine update at epoch {epoch+1}: "
                f"{n_quarantined}/{self.dataset_size} samples quarantined "
                f"({n_quarantined/self.dataset_size*100:.2f}%)"
            )

    def get_stats(self) -> dict:
        """Return quarantine statistics for logging."""
        scored = (self.update_counts > 0).sum().item()
        return {
            "quarantine_count": int(self.quarantined.sum().item()),
            "quarantine_ratio": self.quarantined.float().mean().item(),
            "scored_samples": scored,
            "mean_ema_score": self.ema_scores[self.update_counts > 0].mean().item() if scored > 0 else 0.0,
        }
