"""Pydantic schemas for run metrics and forensic evaluation."""

from pydantic import BaseModel, Field


class RunMetrics(BaseModel):
    """Schema for a single experiment run's metrics."""

    # Experiment identification
    dataset: str
    backbone: str
    ssl_method: str = "simclr"
    defense_mode: str = "none"
    attack_family: str = "none"
    poison_ratio: float = 0.0
    seed: int = 42

    # Utility metrics
    linear_probe_acc: float | None = None
    finetune_acc: float | None = None

    # Security metrics
    asr: float | None = None
    clean_acc_under_attack: float | None = None

    # Systems metrics
    train_time_per_epoch: float | None = None
    overhead_pct: float | None = None
    total_train_time: float | None = None
    peak_memory_mb: float | None = None

    # Training summary
    final_train_loss: float | None = None
    epochs_completed: int = 0


class ForensicMetrics(BaseModel):
    """Schema for forensic evaluation when ground-truth poison indices are known."""

    roc_auc: float | None = None
    pr_auc: float | None = None
    top_k_recall: dict[str, float] = Field(default_factory=dict)
    mean_clean_score: float | None = None
    mean_poison_score: float | None = None
    score_separation: float | None = None


class EpochMetrics(BaseModel):
    """Per-epoch training metrics."""

    epoch: int
    train_loss: float
    lr: float
    epoch_time: float

    # PRCL-specific (optional)
    mean_suspicion: float | None = None
    std_suspicion: float | None = None
    max_suspicion: float | None = None
    quarantine_count: int | None = None
    mean_pos_weight: float | None = None
    mean_neg_weight: float | None = None
