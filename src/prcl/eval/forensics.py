"""Forensic evaluation — measure how well PCF scores separate clean vs. poisoned samples."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from prcl.integritysuite.schemas import ForensicMetrics


def evaluate_forensics(
    suspicion_scores: np.ndarray,
    poison_mask: np.ndarray,
    top_k_fractions: list[float] | None = None,
) -> ForensicMetrics:
    """Evaluate forensic quality of suspicion scores against ground truth.

    Args:
        suspicion_scores: Per-sample suspicion scores (higher = more suspicious).
        poison_mask: Boolean array, True for actually poisoned samples.
        top_k_fractions: Fractions of dataset to check for recall (e.g., [0.01, 0.05, 0.1]).

    Returns:
        ForensicMetrics with ROC-AUC, PR-AUC, top-k recall, and score statistics.
    """
    if top_k_fractions is None:
        top_k_fractions = [0.01, 0.05, 0.1, 0.2]

    n = len(suspicion_scores)
    n_poison = int(poison_mask.sum())

    if n_poison == 0 or n_poison == n:
        return ForensicMetrics()

    # ROC-AUC and PR-AUC
    roc_auc = float(roc_auc_score(poison_mask, suspicion_scores))
    pr_auc = float(average_precision_score(poison_mask, suspicion_scores))

    # Top-k recall: of the top k% most suspicious, how many are actual poisons?
    ranked = np.argsort(-suspicion_scores)  # highest suspicion first
    top_k_recall = {}
    for frac in top_k_fractions:
        k = max(1, int(n * frac))
        top_k_set = set(ranked[:k].tolist())
        poison_in_top_k = sum(1 for i in top_k_set if poison_mask[i])
        recall = poison_in_top_k / n_poison
        top_k_recall[f"top_{frac:.2f}"] = round(recall, 4)

    # Score statistics
    clean_mask = ~poison_mask
    mean_clean = float(suspicion_scores[clean_mask].mean()) if clean_mask.any() else None
    mean_poison = float(suspicion_scores[poison_mask].mean()) if poison_mask.any() else None
    separation = (mean_poison - mean_clean) if (mean_clean is not None and mean_poison is not None) else None

    return ForensicMetrics(
        roc_auc=round(roc_auc, 4),
        pr_auc=round(pr_auc, 4),
        top_k_recall=top_k_recall,
        mean_clean_score=round(mean_clean, 4) if mean_clean is not None else None,
        mean_poison_score=round(mean_poison, 4) if mean_poison is not None else None,
        score_separation=round(separation, 4) if separation is not None else None,
    )
