"""PRCL Defense — main coordinator that integrates PCF, robust loss, and quarantine.

This module provides the `PRCLDefense` class that the training loop calls at
each batch and epoch boundary. It orchestrates:
1. Probe view generation and PCF scoring
2. Positive and negative reweighting via WeightedInfoNCELoss
3. Optional quarantine management
4. Metrics collection for logging
"""

import logging

import torch

from prcl.defenses.prcl.pcf import build_pcf_scorer
from prcl.defenses.prcl.quarantine import QuarantineManager
from prcl.defenses.prcl.robust_loss import WeightedInfoNCELoss

logger = logging.getLogger(__name__)


class PRCLDefense:
    """PRCL defense coordinator — plug-and-play defense for SimCLR training.

    Usage in train loop:
        defense = PRCLDefense(cfg, device)
        for (view1, view2), indices in loader:
            h, z = model(torch.cat([view1, view2]))
            loss, stats = defense.compute_defended_loss(
                h=h, z=z, indices=indices,
                view1=view1, view2=view2, model=model, criterion=criterion
            )
    """

    def __init__(self, cfg, device: torch.device, dataset_size: int = 50000):
        self.cfg = cfg
        self.device = device

        # PCF scorer
        self.pcf_scorer = build_pcf_scorer(cfg)

        # Robust loss
        robust_cfg = cfg.defense.robust
        self.robust_loss = WeightedInfoNCELoss(
            temperature=cfg.ssl.temperature,
            mode=robust_cfg.mode,
            lambda_pos=robust_cfg.lambda_pos,
            lambda_neg=robust_cfg.lambda_neg if cfg.defense.negatives.reweight else 0.0,
            w_min=robust_cfg.w_min,
            trim_alpha=robust_cfg.trim_alpha,
            grad_cap_enabled=robust_cfg.grad_cap_enabled,
            grad_cap_value=robust_cfg.grad_cap_value,
        )

        # Quarantine (optional)
        self.quarantine = None
        if cfg.defense.quarantine.enabled:
            self.quarantine = QuarantineManager(
                dataset_size=dataset_size,
                threshold=cfg.defense.quarantine.threshold,
                percentile=cfg.defense.quarantine.percentile,
                reevaluate_every=cfg.defense.quarantine.reevaluate_every,
            )

    def compute_defended_loss(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        indices: torch.Tensor,
        view1: torch.Tensor,
        view2: torch.Tensor,
        model: torch.nn.Module,
        criterion,
    ) -> tuple[torch.Tensor, dict]:
        """Compute PRCL-defended contrastive loss.

        Args:
            h: Backbone representations (2N, feat_dim).
            z: Projected features (2N, proj_dim).
            indices: Sample indices (N,).
            view1, view2: Augmented views (N, C, H, W).
            model: SimCLR model (for encoding probe views).
            criterion: Vanilla loss (used as fallback if PCF disabled).

        Returns:
            (loss, stats_dict)
        """
        batch_size = z.shape[0] // 2
        stats = {}

        # --- Step 1: Compute PCF suspicion scores ---
        if self.cfg.defense.pcf.enabled:
            # Use view1 as the "clean" reference for probing
            # Representations from h[:batch_size] correspond to view1
            clean_h = h[:batch_size].detach()
            suspicion_scores = self.pcf_scorer.compute_scores(
                clean_images=view1,
                encoder=model.backbone,
                clean_representations=clean_h,
            )

            stats["mean_suspicion"] = suspicion_scores.mean().item()
            stats["std_suspicion"] = suspicion_scores.std().item()
            stats["max_suspicion"] = suspicion_scores.max().item()
            stats["min_suspicion"] = suspicion_scores.min().item()
        else:
            suspicion_scores = None

        # --- Step 2: Quarantine check ---
        if self.quarantine is not None and suspicion_scores is not None:
            self.quarantine.update_scores(indices, suspicion_scores)
            quarantine_mask = self.quarantine.is_quarantined(indices)

            if quarantine_mask.any():
                # Set quarantined sample scores to maximum (they'll get near-zero weight)
                suspicion_scores[quarantine_mask] = 1.0
                stats["quarantine_count"] = quarantine_mask.sum().item()
            else:
                stats["quarantine_count"] = 0

        # --- Step 3: Compute robust loss ---
        if suspicion_scores is not None and self.cfg.defense.robust.mode != "none":
            loss, loss_stats = self.robust_loss(z, suspicion_scores)
            stats.update(loss_stats)
        else:
            loss = criterion(z)
            stats["mean_pos_weight"] = 1.0
            stats["mean_neg_weight"] = 1.0

        return loss, stats

    def on_epoch_start(self, epoch: int):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, epoch: int):
        """Called at the end of each epoch."""
        if self.quarantine is not None:
            self.quarantine.on_epoch_end(epoch)


def build_prcl_defense(cfg, device: torch.device, dataset_size: int = 50000) -> PRCLDefense:
    """Factory function to build a PRCL defense from config."""
    return PRCLDefense(cfg, device, dataset_size)
