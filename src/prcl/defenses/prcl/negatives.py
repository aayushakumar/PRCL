"""Negative-set sanitation — independent control for suspicious negative reweighting.

This module provides the logic for computing per-negative weights based on
suspicion scores, independent of positive reweighting. Per PRD §11.5, negative
sanitation should be toggleable independently from positive pair reweighting.
"""

import torch


def compute_negative_weights(
    suspicion_scores: torch.Tensor,
    lambda_neg: float = 0.4,
    w_min: float = 0.2,
) -> torch.Tensor:
    """Compute per-sample negative weights from suspicion scores.

    Args:
        suspicion_scores: Per-sample suspicion q(x) ∈ [0,1] of shape (N,).
        lambda_neg: Strength of negative downweighting.
        w_min: Minimum weight to prevent complete exclusion.

    Returns:
        Weights of shape (N,) where lower suspicion → higher weight.
    """
    scores = suspicion_scores.detach()
    weights = torch.clamp(1.0 - lambda_neg * scores, min=w_min)
    return weights


def build_negative_weight_matrix(
    suspicion_scores: torch.Tensor,
    lambda_neg: float = 0.4,
    w_min: float = 0.2,
) -> torch.Tensor:
    """Build a (2N, 2N) negative weight matrix for the contrastive loss.

    For each pair (i, j), the weight of j as a negative for i depends on q_j.
    Scores are expanded from (N,) to (2N,) to cover both views.

    Args:
        suspicion_scores: Per-sample suspicion q(x) ∈ [0,1] of shape (N,).
        lambda_neg: Negative reweighting strength.
        w_min: Minimum weight floor.

    Returns:
        Weight matrix of shape (2N, 2N).
    """
    q = suspicion_scores.detach()
    q_2n = q.repeat_interleave(2)
    n_2n = q_2n.shape[0]

    # Each column j has weight based on q_j (as a negative for any row i)
    neg_weights = torch.clamp(
        1.0 - lambda_neg * q_2n.unsqueeze(0).expand(n_2n, -1),
        min=w_min,
    )
    return neg_weights
