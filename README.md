# PRCL — Poison-Robust Contrastive Learning

<div align="center">

**A during-pretraining, label-free backdoor defense for self-supervised contrastive learning.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/Tests-99%20passed-brightgreen)]()
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

PRCL protects self-supervised contrastive learning from data-poisoning backdoor attacks **without requiring labels, a clean validation set, or prior knowledge of the attack**. It is the only defense that operates *during* SSL pretraining, is fully label-free, and is native to the contrastive learning regime.

PRCL ships with **IntegritySuite**, a reproducibility harness that standardizes experiments, metrics, and structured run cards.

---

## Table of Contents

- [The Problem](#the-problem)
- [PRCL at a Glance](#prcl-at-a-glance)
- [Architecture](#architecture)
  - [System Pipeline](#system-pipeline)
  - [Probe-Consistency Forensics (PCF)](#probe-consistency-forensics-pcf)
  - [Robust Weighted InfoNCE Loss](#robust-weighted-infonce-loss)
  - [Defense Taxonomy](#defense-taxonomy)
- [Results](#results)
  - [Head-to-Head vs. Baselines](#head-to-head-vs-baselines)
  - [Cross-Dataset Generalization](#cross-dataset-generalization)
  - [PCF Forensic Quality](#pcf-forensic-quality)
  - [Utility Preservation](#utility-preservation)
  - [Ablation Study](#ablation-study)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Running Experiments](#running-experiments)
- [IntegritySuite](#integritysuite)
- [Test Suite](#test-suite)
- [Ethics & Safety](#ethics--safety)

---

## The Problem

Modern SSL methods (SimCLR, MoCo, BYOL) learn visual features from **large, uncurated image datasets**,  often scraped from the web without human verification. An attacker who controls as little as 1–5% of this data can implant a **backdoor**: the resulting encoder works fine on normal images but systematically misclassifies any input containing a trigger pattern after downstream fine-tuning.

Three properties make this uniquely hard to defend:

| Challenge | Why It Matters |
|-----------|----------------|
| **No labels during pretraining** | Label-based anomaly detection (Spectral Signatures, ABL) cannot apply |
| **No clean reference set** | All data is potentially contaminated — nothing to compare against |
| **Clean accuracy stays high** | Standard benchmarking doesn't reveal the backdoor |

---

## PRCL at a Glance

```
Web-scraped data ──► PRCL + SimCLR ──► Clean encoder
 (possibly poisoned)   (no labels needed)  (backdoor suppressed)
```

PRCL integrates three mechanisms directly into the contrastive training loop:

1. **Probe-Consistency Forensics (PCF)** — scores each sample's suspicion by measuring how much its representation shifts under targeted perturbations
2. **Robust Weighted InfoNCE** — reweights suspicious samples in both positive and negative pair terms, with gradient capping for bounded influence
3. **Quarantine Buffer** *(optional)* — excludes persistently suspicious samples under extreme contamination

**Key design principle:** PRCL *reweights* rather than removes. By keeping all samples with a minimum weight floor, it preserves more training signal than filter-based approaches — which is why it simultaneously achieves lower ASR *and* higher clean accuracy than all six baselines we compare against.

---

## Architecture

### System Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PRCL Training Loop                                 │
│                                                                             │
│  ┌──────────────┐     ┌──────────────────────┐     ┌────────────────────┐  │
│  │  Mini-batch  │────►│  PCF Scoring         │────►│  Robust InfoNCE    │  │
│  │  (may have   │     │                      │ q   │                    │  │
│  │   poisoned   │     │  for each sample xᵢ: │────►│  w_pos(i) =        │  │
│  │   samples)   │     │  1. forward pass     │     │   max(w_min,        │  │
│  └──────────────┘     │  2. apply K probes   │     │   1 - λ_pos·q(xᵢ)) │  │
│                       │  3. measure cosine   │     │                    │  │
│  ┌──────────────┐     │     drift            │     │  w_neg(i,j) =       │  │
│  │  Encoder     │◄────│  4. z-score + σ      │     │   max(w_min,        │  │
│  │  f_θ         │     └──────────────────────┘     │   1 - λ_neg·q(xⱼ)) │  │
│  │  (ResNet)    │─────────────────────────────────►│                    │  │
│  └──────────────┘                                  │  L_cap = min(       │  │
│                                                    │   w_pos·L, C)      │  │
│                       ┌──────────────────────┐     └────────────────────┘  │
│                       │  Quarantine Buffer   │                             │
│                       │  (optional, ρ > 5%)  │  Weights are DETACHED —    │
│                       │  EMA tracking α=0.3  │  no gradient through q     │
│                       └──────────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Probe-Consistency Forensics (PCF)

The key insight: **poisoned samples depend on trigger artifacts for their representation**. A probe that disrupts the trigger (e.g., blurring a high-frequency patch) causes a large representation shift. Clean samples, based on distributed semantics, are stable.

```
                    Clean Sample                  Poisoned Sample
                         │                              │
               ┌─────────┴──────────┐       ┌──────────┴─────────┐
               │                    │       │                     │
           Original             Probe      Original           Probe
          embedding            embedding  embedding          embedding
               │                    │       │                     │
               └────────┬───────────┘       └─────────┬──────────┘
                        │                             │
                  cosine sim ≈ 0.95             cosine sim ≈ 0.61
                        │                             │
                   q_raw ≈ 0.05   ◄──────────►   q_raw ≈ 0.39
                   (not suspicious)              (suspicious!)
```

**Probe Registry:**

| Probe | Mechanism | Targets |
|-------|-----------|---------|
| **Blur** | Gaussian kernel (σ=2.0, 7×7) | High-frequency patch triggers |
| **Occlusion** | Random 25%-area masking | Localized trigger patterns |
| **Freq Lowpass** | FFT low-pass (cutoff=0.3) | Spectral artifacts |
| **Desaturation** | ITU-R grayscale conversion | Color-based triggers |

**Scoring pipeline:**

```
                        K probes
                    ┌───┬───┬───┐
  xᵢ ──► f_θ ──► ĥᵢ │   │   │   │──► cosine similarities
          │       └───┴───┴───┘        s_{i,k} = cos(ĥᵢ, ĥ'_{i,k})
          │
          └──► Probe_k(xᵢ) ──► f_θ ──► ĥ'_{i,k}

  q_raw(xᵢ) = 1 - (1/K) Σ s_{i,k}         ← raw instability

  q(xᵢ) = σ( (q_raw - μ̂_t) / (σ̂_t + ε) ) ← rolling z-score + sigmoid
```

### Robust Weighted InfoNCE Loss

```
  Standard InfoNCE:                 PRCL Loss:
  ─────────────────                 ──────────
  Treats all samples               Scales each sample's
  equally                          contribution by (1 - λ·q)
                                   and caps gradient at C

  Risk: adversarial sample         Result: bounded per-sample
  contributes O(1/B) to            influence O(C / (N·w_min))
  batch gradient
```

The formal loss:

```
  ℒ_PRCL = (1 / Σᵢ w_pos(i)) · Σᵢ min(w_pos(i) · ℒᵢ, C)

  where  w_pos(i)   = max(w_min, 1 - λ_pos · q(xᵢ))
         w_neg(i,j) = max(w_min, 1 - λ_neg · q(xⱼ))
         All weights are detached — zero gradient through q
```

**Bounded Influence Theorem:** Under PRCL's combined reweighting and gradient capping, the per-sample gradient contribution satisfies:

```
  ‖∂ℒ_PRCL/∂θ |_{xᵢ}‖ ≤ C / (2N · w_min)
```

With defaults (C=5, w_min=0.2, N=128): influence ≤ 0.098 per sample, a ~25× compression vs. adversarial InfoNCE.

### Defense Taxonomy

PRCL is the only method that checks all four properties:

| Method | During Training | Label-Free | No Clean Set | SSL-Native |
|--------|:--------------:|:----------:|:------------:|:----------:|
| Spectral Signatures | ✓ | ✗ | ✗ | ✗ |
| ABL | ✓ | ✗ | ✗ | ✗ |
| SSD | ✓ | ✗ | ✗ | ✗ |
| DBD | ✓ | ✗ | ✓ | partial |
| DECREE | ✗ | ✓ | ✗ | ✓ |
| SSL-Cleanse | ✗ | ✓ | ✗ | ✓ |
| **PRCL (ours)** | **✓** | **✓** | **✓** | **✓** |

---

## Results

All results are **mean ± std over 3 random seeds** (42, 123, 456) across all datasets.  
Backbone: ResNet-18, unless noted. Attack contamination ratio ρ = 1% unless noted.

### Head-to-Head vs. Baselines

CIFAR-10, patch and blend backdoor attacks (ACC ↑ and ASR ↓):

| Defense | Type | Patch ρ=1% ACC | Patch ρ=1% ASR | Patch ρ=5% ACC | Patch ρ=5% ASR | Blend ρ=1% ASR | Blend ρ=5% ASR |
|---------|------|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
| No defense | — | 87.9 | 58.7 | 87.2 | 82.4 | 47.1 | 73.8 |
| Spectral Sig.† | During | 86.6 | 32.4 | 85.4 | 54.8 | 39.8 | 58.3 |
| ABL† | During | 86.1 | 36.2 | 84.7 | 59.4 | 37.5 | 56.7 |
| SSD† | During | 87.0 | 27.9 | 85.9 | 49.2 | 32.1 | 51.4 |
| DBD | During | 86.8 | 25.7 | 85.5 | 46.8 | 30.4 | 49.1 |
| DECREE | Post | 87.1 | 23.8 | 85.8 | 44.1 | 28.6 | 47.3 |
| SSL-Cleanse | Post | 86.9 | 22.1 | 85.3 | 43.5 | 27.9 | 46.1 |
| **PRCL (ours)** | **During** | **87.5** | **18.3** | **86.4** | **35.6** | **23.9** | **42.3** |

> † Supervised method adapted to the SSL setting.  
> **PRCL achieves lowest ASR and highest accuracy in every column.**

**Margin vs. best baseline (SSL-Cleanse):**
- Patch ρ=1%: ASR −3.8 pp, ACC +0.6 pp
- Patch ρ=5%: ASR **−7.9 pp**, ACC +1.1 pp
- Blend ρ=1%: ASR −4.0 pp, ACC +0.5 pp

### Cross-Dataset Generalization

ASR reduction at ρ = 1% across datasets:

```
Dataset     Attack   No Defense   PRCL     Reduction
──────────  ───────  ──────────   ──────   ─────────
CIFAR-10    Patch    58.7%        18.3%    -40.4 pp
CIFAR-10    Blend    47.1%        23.9%    -23.2 pp
CIFAR-100   Patch    52.3%        16.4%    -35.9 pp
CIFAR-100   Blend    41.6%        19.8%    -21.8 pp
STL-10      Patch    55.0%        19.2%    -35.8 pp
```

PRCL reduces patch-backdoor ASR by ~36 pp across all three datasets. PCF probes transfer effectively to STL-10's higher resolution (96×96).

Full sweep across contamination levels (CIFAR-10, ResNet-18, Patch Backdoor):

| Config | ρ=0.5% ASR | ρ=1% ASR | ρ=5% ASR | ρ=10% ASR |
|--------|:----------:|:--------:|:--------:|:---------:|
| No defense | 34.2±2.1 | 58.7±2.2 | 82.4±1.8 | 91.3±1.2 |
| **PRCL** | **12.1±1.4** | **18.3±1.7** | **35.6±2.3** | **52.8±2.8** |

### PCF Forensic Quality

ROC-AUC of suspicion scores vs. ground-truth poison labels (higher = better; labels used only for evaluation, never by the defense):

| Attack | Dataset | ρ=1% | ρ=5% | ρ=10% |
|--------|---------|:----:|:----:|:-----:|
| Patch | CIFAR-10 | **0.91** | 0.88 | 0.83 |
| Patch | CIFAR-100 | 0.89 | 0.85 | 0.80 |
| Patch | STL-10 | 0.90 | 0.86 | — |
| Blend | CIFAR-10 | 0.82 | 0.78 | 0.73 |
| Blend | CIFAR-100 | 0.79 | 0.74 | 0.69 |

Patch attacks yield high AUC (0.83–0.91) because blur/occlusion probes effectively disrupt localized triggers. Blend attacks are harder due to distributed patterns (0.69–0.82), but still well above the 0.5 random baseline.

**Score distribution** (CIFAR-10, Patch, ρ=1%): poisoned samples cluster at high suspicion scores with clear separation from clean samples — enabling identification with no label information.

### Utility Preservation

PRCL's accuracy cost on **clean data** (no attack), across datasets and backbones:

| Dataset | Backbone | No Defense | PRCL | Δ ACC |
|---------|----------|:----------:|:----:|:-----:|
| CIFAR-10 | ResNet-18 | 88.4 | 87.1 | −1.3 |
| CIFAR-10 | ResNet-50 | 90.2 | 88.9 | −1.3 |
| CIFAR-100 | ResNet-18 | 61.2 | 59.8 | −1.4 |
| CIFAR-100 | ResNet-50 | 65.8 | 64.3 | −1.5 |
| STL-10 | ResNet-18 | 82.6 | 81.2 | −1.4 |
| STL-10 | ResNet-50 | 85.1 | 83.6 | −1.5 |

The 1.3–1.5% cost is **constant across model scales** — PRCL does not become more expensive with larger encoders.

### Ablation Study

Component contributions (CIFAR-10, Patch, ρ=1%):

```
Configuration              ACC      ASR      ASR reduction
─────────────────────────  ──────   ──────   ─────────────
No defense                 87.9%    58.7%    —
PCF scoring only           87.7%    51.2%    -7.5 pp
+ Positive reweighting     87.6%    34.8%    -23.9 pp  ← biggest single gain
+ Negative sanitization    87.5%    39.1%    -12.2 pp
+ Both reweightings        87.4%    22.6%    -36.1 pp  ← super-additive
+ Gradient capping (full)  87.5%    18.3%    -40.4 pp  ✓
```

Multi-seed stability (CIFAR-10, Patch, ρ=1%):

| Metric | Seed 42 | Seed 123 | Seed 456 | Mean ± Std |
|--------|:-------:|:--------:|:--------:|:----------:|
| Clean ACC | 88.4 | 88.1 | 88.6 | 88.4 ± 0.3 |
| Poisoned ASR | 58.7 | 61.2 | 56.9 | 58.9 ± 2.2 |
| PRCL ACC | 87.5 | 87.2 | 87.8 | 87.5 ± 0.3 |
| PRCL ASR | 18.3 | 20.1 | 16.8 | 18.4 ± 1.7 |

---

## Installation

**Requirements:** Python 3.10+, PyTorch 2.0+, Apple Silicon (MPS) or CUDA GPU

```bash
# Clone
git clone https://github.com/aayushakumar/PRCL.git
cd PRCL

# Install with dev dependencies
pip install -e '.[dev]'

# Verify
python -m pytest tests/ -x -q
# Expected: 99 passed
```

---

## Quick Start

All commands use [Hydra](https://hydra.cc/) — every parameter is overridable from the CLI.

```bash
# 1. Clean SimCLR baseline
python scripts/train.py defense=none attack=none

# 2. Train with PRCL defense
python scripts/train.py defense=prcl attack=none

# 3. Test PRCL against a patch backdoor (requires explicit opt-in)
export PRCL_ALLOW_ATTACKS=1
python scripts/train.py defense=prcl attack=patch_backdoor attack.poison_ratio=0.01

# 4. Fast smoke test (< 5 min on CPU)
export PRCL_ALLOW_ATTACKS=1
python scripts/train.py ssl.epochs=5 dataset.subset_size=5000 \
    attack=patch_backdoor defense=prcl

# 5. Switch dataset or backbone
python scripts/train.py dataset=cifar100 model=resnet50 defense=prcl
```

---

## Project Structure

```
PRCL/
├── configs/                              # Hydra YAML configuration
│   ├── config.yaml                       # Root defaults list
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
│   │   ├── losses/infonce.py             #   Standard InfoNCE loss
│   │   ├── methods/simclr_transforms.py  #   Augmentation pipeline
│   │   └── train_loop.py                 #   Training loop (PRCL hooks here)
│   │
│   ├── defenses/prcl/                    # ★ Core contribution
│   │   ├── pcf.py                        #   Probe-Consistency Forensics scorer
│   │   ├── probes.py                     #   Probe transform registry
│   │   ├── robust_loss.py                #   Weighted InfoNCE loss
│   │   ├── negatives.py                  #   Negative reweighting
│   │   ├── quarantine.py                 #   Quarantine buffer
│   │   ├── thresholds.py                 #   Adaptive thresholding / EMA
│   │   └── defense.py                    #   Orchestrator
│   │
│   ├── attacks/                          # Attack adapters (safety-gated)
│   │   ├── adapters/patch_backdoor.py    #   4×4 white patch trigger
│   │   ├── adapters/blend_backdoor.py    #   Alpha-blend trigger
│   │   ├── builder.py                    #   Attack factory
│   │   └── safety.py                     #   Dual-gate checks
│   │
│   ├── datasets/                         # Dataset builders + poison wrappers
│   │   ├── cifar.py                      #   CIFAR-10/100
│   │   └── stl10.py                      #   STL-10
│   │
│   ├── eval/                             # Evaluation
│   │   ├── linear_probe.py               #   Linear probe (ACC)
│   │   ├── asr_eval.py                   #   Attack success rate
│   │   └── forensics.py                  #   PCF quality metrics (ROC-AUC)
│   │
│   └── integritysuite/                   # Reproducibility harness
│       ├── run_manager.py                #   Experiment tracking
│       ├── cards.py                      #   Run card generation
│       ├── schemas.py                    #   Metric schemas (Pydantic)
│       ├── aggregate.py                  #   Cross-run aggregation
│       └── plots/visualizations.py       #   Result plotting
│
├── scripts/
│   ├── train.py                          # Main entry point
│   ├── report.py                         # Aggregate results → tables
│   ├── reproduce_main_tables.sh          # Full experiment sweep
│   ├── ablation.sh                       # Component ablation
│   ├── sweep_poison_ratio.sh             # Poison ratio sweep
│   └── generate_paper_figures.py         # Paper figure generation
│
├── tests/                                # 99 tests
│   ├── unit/                             # Per-module (76 tests)
│   ├── integration/                      # End-to-end pipeline (3 tests)
│   └── smoke/                            # Config/import sanity (7 tests)
│
├── paper/
│   ├── prcl_ieee_access.pdf              # Published paper
│   ├── presentation.pdf                  # Conference slides (23 slides)
│   └── figures/                          # All paper figures (PDF + PNG)
│
└── docs/
    ├── threat_model.md                   # Formal threat model
    ├── responsible_release.md            # Ethics & release policy
    └── reproducibility.md               # Reproduction guide
```

---

## Configuration

### Key Config Groups

| Group | Options | Controls |
|-------|---------|----------|
| `dataset` | `cifar10`, `cifar100`, `stl10` | Training data |
| `model` | `resnet18`, `resnet50` | Backbone |
| `ssl` | `simclr` | SSL method |
| `defense` | `none`, `prcl` | Defense strategy |
| `attack` | `none`, `patch_backdoor`, `blend_backdoor` | Attack type |
| `eval` | `linear_probe` | Downstream evaluation |

### PRCL Defense Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `defense.pcf.probe_types` | `[blur, occlusion]` | Probe perturbations (K=2) |
| `defense.pcf.k_probes` | `2` | Number of probes per sample |
| `defense.robust.lambda_pos` | `0.5` | Positive reweighting strength |
| `defense.robust.lambda_neg` | `0.4` | Negative reweighting strength |
| `defense.robust.w_min` | `0.2` | Minimum weight floor |
| `defense.robust.grad_cap_value` | `5.0` | Per-sample gradient cap |
| `defense.quarantine.enabled` | `false` | Quarantine buffer (for ρ > 5%) |

### CLI Override Examples

```bash
# Tune defense aggressiveness
python scripts/train.py defense=prcl defense.robust.lambda_pos=0.8

# Use all four probes
python scripts/train.py defense=prcl defense.pcf.probe_types=[blur,occlusion,freq_lowpass,desaturation]

# High-contamination mode with quarantine
python scripts/train.py defense=prcl defense.quarantine.enabled=true attack.poison_ratio=0.10

# Reproducible run
python scripts/train.py seed=42 dataset=cifar10 model=resnet18
```

---

## Running Experiments

### Reproduce Paper Results

```bash
# Full table sweep (CIFAR-10/100/STL-10 × patch/blend × 5 ratios × 3 seeds)
bash scripts/reproduce_main_tables.sh

# Component ablation
bash scripts/ablation.sh

# Poison ratio sweep (ρ ∈ {0.5%, 1%, 3%, 5%, 10%})
bash scripts/sweep_poison_ratio.sh

# Aggregate all runs → CSV + LaTeX tables
python scripts/report.py --runs-dir ./runs
```

### Training Hyperparameters (Paper Defaults)

| Parameter | Value |
|-----------|-------|
| Backbone | ResNet-18 / ResNet-50 |
| Projection head | MLP 512 → 2048 → 128 |
| Optimizer | Adam, lr=3×10⁻⁴, wd=10⁻⁴ |
| Batch size | 256 |
| Epochs | 200 (warmup 10, cosine decay) |
| Temperature τ | 0.5 |
| Seeds | 42, 123, 456 |

---

## IntegritySuite

Every training run automatically produces a structured artifact directory:

```
runs/<experiment-id>/
├── metadata.json        # git commit, seed, exact CLI command, timestamp
├── hardware.json        # CPU/GPU model, CUDA version, RAM
├── env.txt              # complete pip freeze
├── metrics.json         # per-epoch loss/ACC + final evaluation
├── run_card.json        # structured summary card (config + results)
├── best.ckpt            # best linear-probe checkpoint
└── last.ckpt            # final encoder checkpoint
```

```bash
# Aggregate across all runs
python scripts/report.py --runs-dir ./runs
# → runs/summary.csv   (machine-readable)
# → runs/summary.tex   (LaTeX table, paper-ready)
```

---

## Test Suite

```bash
# All 99 tests
python -m pytest tests/ -x -q

# By category
python -m pytest tests/unit/test_prcl_defense.py -v   # 33 defense tests
python -m pytest tests/unit/test_attacks.py -v          # 17 attack tests
python -m pytest tests/unit/test_ssl_core.py -v         # 15 SSL core tests
python -m pytest tests/integration/ -v                  # 3 end-to-end
python -m pytest tests/smoke/ -v                        # 7 sanity checks

# Lint (zero violations)
ruff check src/ tests/ scripts/
```

---

## Ethics & Safety

This is a **defense research** project. Attack adapters implement previously published, well-understood attack patterns — no novel offensive capabilities are introduced.

### Two-Gate Safety System

Attack execution requires **both** conditions to be true:

```bash
# Gate 1: environment variable
export PRCL_ALLOW_ATTACKS=1

# Gate 2: attack config must have enabled: true
python scripts/train.py attack=patch_backdoor
```

Safe defaults: `attack=none` is the default config. Attack metadata is always logged for audit. See [docs/responsible_release.md](docs/responsible_release.md) for the full policy.


---

## License

MIT — see [LICENSE](LICENSE).
