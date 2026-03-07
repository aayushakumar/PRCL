"""Structured config dataclasses for Hydra schema validation."""

from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    name: str = "cifar10"
    data_dir: str = "./data"
    num_workers: int = 4
    subset_size: int | None = None  # None = use full dataset


@dataclass
class ModelConfig:
    backbone: str = "resnet18"
    pretrained: bool = False
    projection_dim: int = 128
    projection_hidden_dim: int = 2048


@dataclass
class SSLConfig:
    method: str = "simclr"
    epochs: int = 200
    batch_size: int = 256
    temperature: float = 0.5
    optimizer: str = "adam"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    use_amp: bool = False


@dataclass
class PCFConfig:
    enabled: bool = True
    stat: str = "probe_alignment"  # probe_alignment | neighbor_overlap
    k_probes: int = 2
    probe_types: list[str] = field(default_factory=lambda: ["blur", "occlusion"])
    normalize: str = "rolling_zscore"  # rolling_zscore | batch_minmax | none
    neighbor_k: int = 10  # for neighbor_overlap


@dataclass
class RobustConfig:
    mode: str = "soft_weight"  # none | soft_weight | trim | soft_weight+grad_cap
    lambda_pos: float = 0.5
    lambda_neg: float = 0.4
    w_min: float = 0.2
    trim_alpha: float = 0.0
    grad_cap_enabled: bool = True
    grad_cap_value: float = 5.0


@dataclass
class NegativesConfig:
    reweight: bool = True
    false_negative_correction: bool = False


@dataclass
class QuarantineConfig:
    enabled: bool = False
    threshold: float = 0.95
    reevaluate_every: int = 5
    percentile: float = 0.99


@dataclass
class DefenseConfig:
    name: str = "none"  # none | prcl
    enabled: bool = False
    pcf: PCFConfig = field(default_factory=PCFConfig)
    robust: RobustConfig = field(default_factory=RobustConfig)
    negatives: NegativesConfig = field(default_factory=NegativesConfig)
    quarantine: QuarantineConfig = field(default_factory=QuarantineConfig)


@dataclass
class AttackConfig:
    name: str = "none"  # none | patch_backdoor | blend_backdoor
    enabled: bool = False
    poison_ratio: float = 0.01
    target_class: int = 0
    # Patch-specific
    patch_size: int = 4
    patch_position: str = "bottom_right"  # bottom_right | random
    # Blend-specific
    blend_alpha: float = 0.1


@dataclass
class EvalConfig:
    method: str = "linear_probe"  # linear_probe | finetune
    epochs: int = 100
    batch_size: int = 256
    lr: float = 0.1
    weight_decay: float = 0.0
    optimizer: str = "sgd"
    schedule: str = "cosine"


@dataclass
class LoggingConfig:
    log_every_n_steps: int = 50
    save_checkpoint_every: int = 50
    use_tensorboard: bool = False
    use_wandb: bool = False
    wandb_project: str = "prcl"


@dataclass
class ReproducibilityConfig:
    deterministic: bool = True
    benchmark: bool = False


@dataclass
class PRCLConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ssl: SSLConfig = field(default_factory=SSLConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
    seed: int = 42
    run_dir: str = "./runs"
