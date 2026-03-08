# PRCL — Poison-Robust Contrastive Learning

A defense framework that protects self-supervised contrastive learning (SSL) from data poisoning and backdoor attacks — **without needing labels, a clean validation set, or knowledge of the attack**.

PRCL ships with **IntegritySuite**, an evaluation harness that standardizes experiments, metrics, and reporting for SSL backdoor defense research.

```
Scraped dataset ──► SimCLR + PRCL ──► Clean encoder
  (may contain                         (backdoor behavior
   poisoned data)                       is suppressed)
```

---

## Table of Contents

- [The Problem](#the-problem)
- [How PRCL Works](#how-prcl-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Running Experiments](#running-experiments)
- [Evaluation](#evaluation)
- [IntegritySuite](#integritysuite)
- [Test Suite](#test-suite)
- [Documentation](#documentation)
- [Ethics & Safety](#ethics--safety)
- [License](#license)

---

## The Problem

Modern SSL methods (SimCLR, MoCo, BYOL) learn visual features from **large unlabeled image datasets** — often scraped from the web. An attacker who can slip even a small fraction (1–5%) of poisoned images into this data can implant a **backdoor**: the resulting encoder works fine on normal images, but systematically misclassifies inputs containing a trigger pattern after downstream fine-tuning.

This is hard to defend against because:

1. **No labels exist** during pretraining — you can't use label-based anomaly detection
2. **No clean reference set** is available — all your data is potentially contaminated
3. **Clean accuracy stays high** — standard evaluation won't reveal the problem

PRCL addresses all three challenges.

---

## How PRCL Works

PRCL adds three defense layers to standard contrastive SSL training:

### 1. Probe-Consistency Forensics (PCF)

PCF computes a **suspicion score** `q(x) ∈ [0, 1]` for every training sample by testing how stable its representation is under targeted perturbations:

| Probe | What It Does | What It Catches |
|-------|-------------|-----------------|
| **Blur** | Gaussian blur (σ=2.0) | High-frequency patch triggers |
| **Occlusion** | Random 25%-area masking | Localized trigger patterns |
| **Freq Lowpass** | FFT low-pass filter | Spectral artifacts |
| **Desaturation** | Convert to grayscale | Color-based triggers |

**Why it works:** A poisoned sample's representation depends heavily on the trigger artifact. When a probe disrupts that artifact, the representation shifts more than it would for a clean sample. PCF measures this instability.

### 2. Robust Weighted InfoNCE Loss

Instead of the standard contrastive loss that treats all samples equally, PRCL **reweights** both positive and negative terms using PCF scores:

- **Suspicious positives** → reduced weight (so the encoder learns less from them)
- **Suspicious negatives** → reduced influence (so they can't warp the representation space)
- **Gradient capping** → bounds the maximum influence of any single sample

The key property: all weights are **detached** — no gradient flows through the suspicion scores, keeping training stable.

### 3. Quarantine Buffer (Optional)

For high-contamination scenarios: samples with persistently high suspicion scores are temporarily excluded from training and periodically re-evaluated. Disabled by default.

---

## Installation

**Requirements:** Python 3.10+, PyTorch 2.0+

```bash
# Clone the repo
git clone https://github.com/aayushakumar/PRCL.git
cd PRCL

# Install (editable mode with dev dependencies)
pip install -e '.[dev]'

# Verify everything works
python -m pytest tests/ -x -q
# Expected: 99 passed
```

---

## Quick Start

All commands use [Hydra](https://hydra.cc/) for configuration. Override any parameter from the command line.

### 1. Clean SimCLR Baseline (no attack, no defense)

```bash
python scripts/train.py defense=none attack=none
```

### 2. Train with PRCL Defense

```bash
python scripts/train.py defense=prcl attack=none
```

### 3. Test Against a Backdoor Attack

```bash
# Attacks require explicit opt-in (see Ethics section)
export PRCL_ALLOW_ATTACKS=1

# Poisoned training WITHOUT defense
python scripts/train.py defense=none attack=patch_backdoor attack.poison_ratio=0.01

# Poisoned training WITH PRCL defense
python scripts/train.py defense=prcl attack=patch_backdoor attack.poison_ratio=0.01
```

### 4. Smoke Test (fast, < 5 min on CPU)

```bash
export PRCL_ALLOW_ATTACKS=1
python scripts/train.py ssl.epochs=5 dataset.subset_size=5000 \
    attack=patch_backdoor defense=prcl
```

---

## Project Structure

```
PRCL/
├── configs/                              # Hydra YAML configuration
│   ├── config.yaml                       # Top-level defaults list
│   ├── attack/                           # none, patch_backdoor, blend_backdoor
│   ├── dataset/                          # cifar10, cifar100, stl10
│   ├── defense/                          # none, prcl
│   ├── eval/                             # linear_probe
│   ├── model/                            # resnet18, resnet50
│   └── ssl/                              # simclr
│
├── src/prcl/                             # Main Python package
│   ├── ssl/                              # SimCLR implementation
│   │   ├── backbones/resnet.py           #   ResNet-18/50 encoder
│   │   ├── heads/projection.py           #   MLP projection head
│   │   ├── losses/infonce.py             #   InfoNCE contrastive loss
│   │   ├── methods/simclr_transforms.py  #   Augmentation pipeline
│   │   └── train_loop.py                 #   Training loop
│   │
│   ├── defenses/prcl/                    # PRCL defense (core contribution)
│   │   ├── pcf.py                        #   Probe-Consistency Forensics scorer
│   │   ├── probes.py                     #   Probe transform registry
│   │   ├── robust_loss.py                #   Weighted InfoNCE loss
│   │   ├── negatives.py                  #   Negative reweighting
│   │   ├── quarantine.py                 #   Quarantine buffer
│   │   ├── thresholds.py                 #   Adaptive thresholding
│   │   └── defense.py                    #   Orchestrator (ties it all together)
│   │
│   ├── attacks/                          # Attack adapters (safety-gated)
│   │   ├── adapters/patch_backdoor.py    #   4×4 patch trigger
│   │   ├── adapters/blend_backdoor.py    #   Alpha-blended trigger
│   │   ├── builder.py                    #   Attack factory
│   │   └── safety.py                     #   Dual-gate safety checks
│   │
│   ├── datasets/                         # Dataset builders
│   │   ├── cifar.py                      #   CIFAR-10/100 + poison wrapper
│   │   └── stl10.py                      #   STL-10
│   │
│   ├── eval/                             # Evaluation modules
│   │   ├── linear_probe.py               #   Linear probe accuracy
│   │   ├── asr_eval.py                   #   Attack success rate
│   │   └── forensics.py                  #   PCF quality metrics
│   │
│   ├── integritysuite/                   # IntegritySuite harness
│   │   ├── run_manager.py                #   Experiment tracking
│   │   ├── cards.py                      #   Run card generation
│   │   ├── schemas.py                    #   Metric schemas
│   │   ├── aggregate.py                  #   Cross-run aggregation
│   │   └── plots/visualizations.py       #   Result plotting
│   │
│   └── config_schema.py                  # Pydantic config validation
│
├── scripts/                              # Entry points
│   ├── train.py                          #   Main training script
│   ├── report.py                         #   Aggregate results report
│   ├── sanity_check.py                   #   Quick validation
│   ├── reproduce_main_tables.sh          #   Full experiment sweep
│   ├── ablation.sh                       #   Component ablation
│   └── sweep_poison_ratio.sh             #   Poison ratio sweep
│
├── tests/                                # 99 tests
│   ├── unit/                             #   Per-module tests
│   ├── integration/                      #   End-to-end pipeline tests
│   └── smoke/                            #   Config/import sanity tests
│
├── notebooks/
│   └── colab_train.ipynb                 # Google Colab notebook
│
├── docs/
│   ├── threat_model.md                   # Formal threat model
│   ├── responsible_release.md            # Ethics & release policy
│   ├── reproducibility.md                # Reproduction guide
│   ├── PRCL_Paper.md                     # Full research paper (markdown)
│   └── PRCL_Intermediate_Report.tex      # LaTeX intermediate report
│
└── pyproject.toml                        # Package metadata & dependencies
```

---

## Configuration

PRCL uses [Hydra](https://hydra.cc/) for configuration management. Every parameter can be overridden from the CLI.

### Key Config Groups

| Group | Options | What It Controls |
|-------|---------|-----------------|
| `dataset` | `cifar10`, `cifar100`, `stl10` | Training data |
| `model` | `resnet18`, `resnet50` | Backbone architecture |
| `ssl` | `simclr` | Self-supervised method |
| `defense` | `none`, `prcl` | Defense strategy |
| `attack` | `none`, `patch_backdoor`, `blend_backdoor` | Attack type |
| `eval` | `linear_probe` | Downstream evaluation |

### PRCL Defense Parameters

| Parameter | Default | What It Does |
|-----------|---------|-------------|
| `defense.pcf.probe_types` | `[blur, occlusion]` | Which probe perturbations to use |
| `defense.pcf.k_probes` | `2` | Number of probes per sample |
| `defense.robust.lambda_pos` | `0.5` | How aggressively to downweight suspicious positives (0=off, 1=max) |
| `defense.robust.lambda_neg` | `0.4` | How aggressively to downweight suspicious negatives |
| `defense.robust.w_min` | `0.2` | Minimum weight floor (prevents zero-ing out any sample) |
| `defense.robust.grad_cap_value` | `5.0` | Per-sample loss cap for gradient bounding |
| `defense.quarantine.enabled` | `false` | Whether to quarantine high-suspicion samples |

### CLI Override Examples

```bash
# Change dataset and backbone
python scripts/train.py dataset=cifar100 model=resnet50

# Tune PRCL aggressiveness
python scripts/train.py defense=prcl defense.robust.lambda_pos=0.3 defense.robust.w_min=0.1

# Change training duration
python scripts/train.py ssl.epochs=100 ssl.batch_size=512

# Set seed for reproducibility
python scripts/train.py seed=42
```

---

## Running Experiments

### Reproduce Main Results

```bash
# Full sweep across datasets, attacks, and poison ratios (requires GPU)
bash scripts/reproduce_main_tables.sh

# Component ablation (PCF only, robust only, full PRCL)
bash scripts/ablation.sh

# Poison ratio sweep (ρ = 0.5%, 1%, 3%, 5%, 10%)
bash scripts/sweep_poison_ratio.sh
```

### Generate Reports

```bash
# Aggregate all runs into a summary report
python scripts/report.py --runs-dir ./runs
```

---

## Evaluation

Every experiment measures three things:

### Clean Accuracy (ACC ↑)

Freeze the trained encoder, attach a linear classifier, train it on labeled data. This tells you whether PRCL preserves useful representations.

```
Encoder (frozen) → Linear head → Classification accuracy
```

### Attack Success Rate (ASR ↓)

Apply the trigger pattern to clean test images and measure how often they're misclassified as the attacker's target class. Lower is better — it means the backdoor is suppressed.

```
Triggered test image → Encoder → Linear head → Classified as target?
```

### PCF Detection Quality (AUC ↑)

Compare PCF suspicion scores against ground-truth poison labels. Higher AUC means PCF is better at identifying which samples are actually poisoned.

---

## IntegritySuite

IntegritySuite is the evaluation infrastructure that makes experiments **reproducible** and **comparable**.

### What It Records Per Run

| Artifact | Purpose |
|----------|---------|
| `metadata.json` | Git commit, seed, exact CLI command, timestamp |
| `hardware.json` | CPU, GPU, CUDA version, memory |
| `env.txt` | Complete `pip freeze` output |
| `metrics.json` | Per-epoch metrics + final evaluation numbers |
| `run_card.json` | Structured experiment summary card |
| `best.ckpt` / `last.ckpt` | Model checkpoints |
| `attack_metadata.json` | Attack configuration (when attacks are enabled) |

### Aggregation

```bash
python scripts/report.py --runs-dir ./runs
```

Generates cross-run comparison tables, leaderboards, and plots.

---

## Test Suite

The codebase has **99 tests** covering every module:

```bash
# Run all tests
python -m pytest tests/ -x -q

# Run specific test categories
python -m pytest tests/unit/test_prcl_defense.py -v    # Defense tests (33)
python -m pytest tests/unit/test_attacks.py -v          # Attack tests (17)
python -m pytest tests/unit/test_ssl_core.py -v         # SSL core tests (15)
python -m pytest tests/integration/ -v                  # End-to-end (3)
python -m pytest tests/smoke/ -v                        # Sanity checks (7)
```

**Lint:** Zero violations with [ruff](https://docs.astral.sh/ruff/).

```bash
ruff check src/ tests/ scripts/
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/threat_model.md](docs/threat_model.md) | Formal attacker/defender models and evaluation tiers |
| [docs/responsible_release.md](docs/responsible_release.md) | Ethics policy and safety mechanisms |
| [docs/reproducibility.md](docs/reproducibility.md) | Step-by-step reproduction guide |
| [docs/PRCL_Paper.md](docs/PRCL_Paper.md) | Full research paper (all sections, placeholder results) |
| [docs/PRCL_Intermediate_Report.tex](docs/PRCL_Intermediate_Report.tex) | LaTeX intermediate report with illustrative results |

---

## Ethics & Safety

This is a **defense research** project. All attack adapters implement well-known, previously published attack patterns and do not introduce novel offensive capabilities.

### Safety Gates

Attack execution requires **two** explicit opt-ins:

```bash
# 1. Environment variable
export PRCL_ALLOW_ATTACKS=1

# 2. Attack config must have enabled: true
python scripts/train.py attack=patch_backdoor  # attack.enabled is set in the YAML
```

Both must be present — if either is missing, attack code will not execute.

### Safe Defaults

- Attacks are **disabled** — `configs/attack/none.yaml` is the default
- No one-click offensive pipeline
- All attack metadata is logged for audit

---

## License

MIT — see [LICENSE](LICENSE).
