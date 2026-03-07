"""Plotting utilities for IntegritySuite reports."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt


def plot_suspicion_histogram(
    suspicion_scores: np.ndarray,
    poison_mask: np.ndarray,
    save_path: Path | str,
    title: str = "Suspicion Score Distribution",
) -> Path:
    """Plot overlapping histograms of suspicion scores for clean vs. poisoned samples."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    clean_scores = suspicion_scores[~poison_mask]
    poison_scores = suspicion_scores[poison_mask]

    bins = np.linspace(0, 1, 50)
    ax.hist(clean_scores, bins=bins, alpha=0.6, label="Clean", color="steelblue", density=True)
    if len(poison_scores) > 0:
        ax.hist(poison_scores, bins=bins, alpha=0.6, label="Poisoned", color="crimson", density=True)

    ax.set_xlabel("Suspicion Score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_training_curves(
    epoch_metrics: list[dict],
    save_path: Path | str,
) -> Path:
    """Plot training loss and optional PRCL metrics over epochs."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [m["epoch"] for m in epoch_metrics]
    losses = [m["train_loss"] for m in epoch_metrics]

    has_suspicion = any(m.get("mean_suspicion") is not None for m in epoch_metrics)
    n_panels = 2 if has_suspicion else 1

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    # Loss curve
    axes[0].plot(epochs, losses, "b-", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Training Loss")

    # Suspicion curve
    if has_suspicion:
        means = [m.get("mean_suspicion", 0) for m in epoch_metrics]
        maxes = [m.get("max_suspicion", 0) for m in epoch_metrics]
        axes[1].plot(epochs, means, "b-", label="Mean", linewidth=1.5)
        axes[1].plot(epochs, maxes, "r--", label="Max", linewidth=1)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Suspicion Score")
        axes[1].set_title("PCF Suspicion Over Training")
        axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_asr_comparison(
    results: dict[str, float],
    save_path: Path | str,
    title: str = "Attack Success Rate Comparison",
) -> Path:
    """Bar chart comparing ASR across defense configurations."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    values = [results[n] * 100 for n in names]

    bars = ax.bar(names, values, color="steelblue", alpha=0.8)
    ax.set_ylabel("ASR (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)

    for bar, val in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
