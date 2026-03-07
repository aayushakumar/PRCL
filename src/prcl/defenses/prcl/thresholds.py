"""Score normalization and thresholding utilities for PCF suspicion scores."""

import torch


class RollingZScoreNormalizer:
    """Normalizes raw scores to [0,1] using rolling mean/std with exponential moving average.

    This ensures suspicion scores are calibrated relative to the recent distribution
    of scores, preventing early-training noise from corrupting later scores.
    """

    def __init__(self, momentum: float = 0.1, eps: float = 1e-8):
        self.momentum = momentum
        self.eps = eps
        self.running_mean = None
        self.running_var = None

    def normalize(self, raw_scores: torch.Tensor) -> torch.Tensor:
        """Normalize raw scores to approximately [0, 1] using rolling z-score + sigmoid.

        Args:
            raw_scores: Tensor of raw suspicion scores (any shape).

        Returns:
            Normalized scores in [0, 1].
        """
        batch_mean = raw_scores.mean().item()
        batch_var = raw_scores.var().item()

        if self.running_mean is None:
            self.running_mean = batch_mean
            self.running_var = max(batch_var, self.eps)
        else:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

        # Z-score normalization
        std = max(self.running_var ** 0.5, self.eps)
        z_scores = (raw_scores - self.running_mean) / std

        # Map to [0, 1] via sigmoid
        normalized = torch.sigmoid(z_scores)
        return normalized

    def reset(self):
        self.running_mean = None
        self.running_var = None


class BatchMinMaxNormalizer:
    """Simple batch-level min-max normalization to [0, 1]."""

    def normalize(self, raw_scores: torch.Tensor) -> torch.Tensor:
        s_min = raw_scores.min()
        s_max = raw_scores.max()
        if s_max - s_min < 1e-8:
            return torch.zeros_like(raw_scores)
        return (raw_scores - s_min) / (s_max - s_min)

    def reset(self):
        pass


def get_normalizer(name: str):
    """Get a score normalizer by name."""
    normalizers = {
        "rolling_zscore": RollingZScoreNormalizer,
        "batch_minmax": BatchMinMaxNormalizer,
    }
    if name == "none":
        return None
    if name not in normalizers:
        raise ValueError(f"Unknown normalizer '{name}'. Available: {list(normalizers)}")
    return normalizers[name]()
