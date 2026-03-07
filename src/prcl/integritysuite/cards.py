"""Run card generation — structured summary for each experiment run."""

import json
from datetime import datetime
from pathlib import Path

from prcl.integritysuite.schemas import ForensicMetrics, RunMetrics


def generate_run_card(
    run_path: Path,
    run_metrics: RunMetrics,
    forensic_metrics: ForensicMetrics | None = None,
    attack_metadata: dict | None = None,
) -> Path:
    """Generate a Markdown run card summarizing one experiment.

    Saved to run_path/cards/run_card.md.
    Returns the path to the created card.
    """
    cards_dir = run_path / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    card_path = cards_dir / "run_card.md"

    lines = [
        "# Run Card",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Run Path:** `{run_path}`",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Dataset | {run_metrics.dataset} |",
        f"| Backbone | {run_metrics.backbone} |",
        f"| SSL Method | {run_metrics.ssl_method} |",
        f"| Defense | {run_metrics.defense_mode} |",
        f"| Attack | {run_metrics.attack_family} |",
        f"| Poison Ratio | {run_metrics.poison_ratio} |",
        f"| Seed | {run_metrics.seed} |",
        "",
    ]

    # Utility metrics
    lines.extend([
        "## Utility Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ])
    if run_metrics.linear_probe_acc is not None:
        lines.append(f"| Linear Probe Acc | {run_metrics.linear_probe_acc:.4f} |")
    if run_metrics.finetune_acc is not None:
        lines.append(f"| Fine-tune Acc | {run_metrics.finetune_acc:.4f} |")
    lines.append("")

    # Security metrics
    if run_metrics.asr is not None:
        lines.extend([
            "## Security Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| ASR | {run_metrics.asr:.4f} |",
        ])
        if run_metrics.clean_acc_under_attack is not None:
            lines.append(f"| Clean Acc (under attack) | {run_metrics.clean_acc_under_attack:.4f} |")
        lines.append("")

    # Forensic metrics
    if forensic_metrics is not None and forensic_metrics.roc_auc is not None:
        lines.extend([
            "## Forensic Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| ROC-AUC | {forensic_metrics.roc_auc:.4f} |",
            f"| PR-AUC | {forensic_metrics.pr_auc:.4f} |",
        ])
        if forensic_metrics.mean_clean_score is not None:
            lines.append(f"| Mean Clean Score | {forensic_metrics.mean_clean_score:.4f} |")
        if forensic_metrics.mean_poison_score is not None:
            lines.append(f"| Mean Poison Score | {forensic_metrics.mean_poison_score:.4f} |")
        if forensic_metrics.score_separation is not None:
            lines.append(f"| Score Separation | {forensic_metrics.score_separation:.4f} |")
        lines.append("")

        if forensic_metrics.top_k_recall:
            lines.extend([
                "### Top-k Recall",
                "",
                "| k | Recall |",
                "|---|--------|",
            ])
            for k, v in sorted(forensic_metrics.top_k_recall.items()):
                lines.append(f"| {k} | {v:.4f} |")
            lines.append("")

    # Attack metadata
    if attack_metadata:
        lines.extend([
            "## Attack Details",
            "",
            "```json",
            json.dumps(attack_metadata, indent=2),
            "```",
            "",
        ])

    # Systems metrics
    lines.extend([
        "## Systems",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ])
    if run_metrics.total_train_time is not None:
        lines.append(f"| Total Train Time | {run_metrics.total_train_time:.1f}s |")
    if run_metrics.train_time_per_epoch is not None:
        lines.append(f"| Time/Epoch | {run_metrics.train_time_per_epoch:.1f}s |")
    if run_metrics.overhead_pct is not None:
        lines.append(f"| Defense Overhead | {run_metrics.overhead_pct:.1f}% |")
    if run_metrics.peak_memory_mb is not None:
        lines.append(f"| Peak Memory | {run_metrics.peak_memory_mb:.0f} MB |")
    lines.append("")

    card_path.write_text("\n".join(lines))

    # Also save structured JSON
    json_path = cards_dir / "run_card.json"
    card_data = {
        "run_metrics": run_metrics.model_dump(),
        "forensic_metrics": forensic_metrics.model_dump() if forensic_metrics else None,
        "attack_metadata": attack_metadata,
    }
    json_path.write_text(json.dumps(card_data, indent=2))

    return card_path
