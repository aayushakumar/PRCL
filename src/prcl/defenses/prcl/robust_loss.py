"""Robust contrastive loss — weighted InfoNCE with positive/negative reweighting.

This is the core optimization component of PRCL. It modifies the standard
InfoNCE loss to reduce the influence of suspicious samples through:

1. Positive reweighting: w_pos(i) = max(w_min, 1 - λ_pos · q_i)
2. Negative reweighting: w_neg(i,k) = max(w_min, 1 - λ_neg · q_k)
3. Loss clipping: cap per-sample loss at a maximum value

Key design choice: all weights are DETACHED — no gradient flows through
suspicion scores back to the encoder. This prevents training instability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedInfoNCELoss(nn.Module):
    """Weighted NT-Xent loss with per-sample positive and negative weighting.

    Supports modes:
        - none: standard InfoNCE (ignores suspicion scores)
        - soft_weight: apply positive + negative reweighting
        - trim: drop top-α suspicious samples entirely
        - soft_weight+grad_cap: reweight + per-sample loss clipping
    """

    def __init__(
        self,
        temperature: float = 0.5,
        mode: str = "soft_weight",
        lambda_pos: float = 0.5,
        lambda_neg: float = 0.4,
        w_min: float = 0.2,
        trim_alpha: float = 0.0,
        grad_cap_enabled: bool = True,
        grad_cap_value: float = 5.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.mode = mode
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        self.w_min = w_min
        self.trim_alpha = trim_alpha
        self.grad_cap_enabled = grad_cap_enabled
        self.grad_cap_value = grad_cap_value

    def forward(
        self,
        z: torch.Tensor,
        suspicion_scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute weighted InfoNCE loss.

        Args:
            z: Projected features of shape (2N, proj_dim) where z[2i] and z[2i+1]
               are the two views of sample i.
            suspicion_scores: Per-sample scores q(x) ∈ [0,1] of shape (N,).
                If None, falls back to standard InfoNCE.

        Returns:
            (loss, stats_dict) where stats_dict contains monitoring values.
        """
        batch_size = z.shape[0] // 2  # N
        z = F.normalize(z, dim=1)

        # Full 2N x 2N similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature

        # Mask self-similarity
        mask_self = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask_self, -9e15)

        # Positive pair indices
        pos_indices = torch.arange(2 * batch_size, device=z.device)
        pos_indices[0::2] += 1
        pos_indices[1::2] -= 1

        pos_sim = sim[torch.arange(2 * batch_size, device=z.device), pos_indices]

        stats = {}

        if suspicion_scores is None or self.mode == "none":
            # Standard InfoNCE
            loss_per_sample = -pos_sim + torch.logsumexp(sim, dim=1)
            loss = loss_per_sample.mean()
            stats["mean_pos_weight"] = 1.0
            stats["mean_neg_weight"] = 1.0
        else:
            # Expand scores from (N,) to (2N,): both views of sample i get score q_i
            q = suspicion_scores.detach()  # CRITICAL: detach to prevent gradient flow
            q_expanded = q.repeat_interleave(2)  # (2N,)

            # --- Positive reweighting ---
            w_pos = torch.clamp(1.0 - self.lambda_pos * q_expanded, min=self.w_min)

            # --- Negative reweighting ---
            if self.lambda_neg > 0:
                # For each row i, the negative weight of column j depends on q_j
                neg_weights = torch.clamp(
                    1.0 - self.lambda_neg * q_expanded.unsqueeze(0).expand(2 * batch_size, -1),
                    min=self.w_min,
                )
                # Zero out self and positive pair weights (they are handled separately)
                neg_weights.masked_fill_(mask_self, 0)
                # Apply log-domain negative weighting
                weighted_sim = sim + torch.log(neg_weights + 1e-10)
            else:
                weighted_sim = sim

            # --- Trimming ---
            if "trim" in self.mode and self.trim_alpha > 0:
                # Identify top-α% most suspicious samples and zero their contribution
                k = max(1, int(batch_size * self.trim_alpha))
                _, trim_indices = q.topk(k)
                trim_mask_2n = torch.zeros(2 * batch_size, dtype=torch.bool, device=z.device)
                trim_mask_2n[trim_indices * 2] = True
                trim_mask_2n[trim_indices * 2 + 1] = True
                w_pos[trim_mask_2n] = 0.0

            # Compute per-sample loss
            log_denom = torch.logsumexp(weighted_sim, dim=1)
            loss_per_sample = -pos_sim + log_denom

            # Apply positive weights
            weighted_loss = w_pos * loss_per_sample

            # --- Loss clipping (grad cap approximation) ---
            if self.grad_cap_enabled and "grad_cap" in self.mode:
                weighted_loss = torch.clamp(weighted_loss, max=self.grad_cap_value)

            # Average over non-trimmed samples
            if w_pos.sum() > 0:
                loss = weighted_loss.sum() / torch.clamp(w_pos.sum(), min=1.0)
            else:
                loss = weighted_loss.mean()

            stats["mean_pos_weight"] = w_pos.mean().item()
            stats["mean_neg_weight"] = neg_weights.mean().item() if self.lambda_neg > 0 else 1.0
            stats["min_pos_weight"] = w_pos.min().item()
            stats["max_suspicion_in_batch"] = q.max().item()

        return loss, stats
