# PRCL + IntegritySuite
**Poison-Resistant Contrastive Pretraining (PRCL)** with **IntegritySuite** — a practical, label-free “default defense” for vision-only contrastive SSL under data poisoning/backdoors, plus a reusable evaluation harness.

> Primary goal: **plug-and-play defense** for SimCLR/MoCo-style SSL that preserves clean utility while reducing backdoor/poison impact during **unlabeled pretraining**.

## Table of Contents
- [Threat model](#threat-model)
- [What PRCL does](#what-prcl-does)
  - [Probe-Consistency Forensics (PCF)](#probe-consistency-forensics-pcf)
  - [Robust contrastive optimization](#robust-contrastive-optimization)
  - [Poison-safe negatives](#poison-safe-negatives)
  - [Quarantine buffer (optional)](#quarantine-buffer-optional)
- [What IntegritySuite provides](#what-integritysuite-provides)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [Repo layout](#repo-layout)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Metrics & reporting](#metrics--reporting)
- [Reproducibility](#reproducibility)
- [Theory (partial guarantees)](#theory-partial-guarantees)
- [Ethics & responsible use](#ethics--responsible-use)
- [Roadmap](#roadmap)
- [Citing](#citing)
- [License](#license)

---

## Why PRCL
Contrastive self-supervised learning (SSL) pretraining often runs on **large, scraped, unlabeled datasets**. In that setting, a small fraction of poisoned samples can implant **backdoor behavior** that survives downstream fine-tuning while keeping clean utility deceptively high.

**Gap this repo targets:** the community lacks a widely adopted, **simple, vision-only “default defense”** that is:
- label-free during pretraining,
- practical and compute-aware,
- evaluated against modern strong SSL poisoning/backdoor attacks,
- shipped with a reusable benchmark harness so results are hard to dismiss.

This repo ships:
1) **PRCL** (defense)  
2) **IntegritySuite** (evaluation harness: attacks + metrics + reporting + reproducibility)


## Threat model
### Defender
- Controls the SSL pipeline: augmentations, sampling, loss, optimization, checkpoints.
- **No labels** for pretraining data.
- Optional: a **small clean anchor set** (unlabeled but curated) can be used in some modes (Semi-supervised?).

### Attacker
- Can inject a small fraction `ρ` of poisoned samples into pretraining corpus.
- Objective: encoder behaves normally on clean data, but misbehaves on triggered inputs downstream.
- May be gray-box or white-box; evaluation supports both where feasible.

> Note: IntegritySuite is built for **defense evaluation**. “Attack runs” require explicit opt-in and are disabled by default (see Ethics section).


## What PRCL does
PRCL is built around a simple idea: **poisoned samples often exhibit abnormal embedding behavior under targeted “probe” perturbations**, and robust optimization can prevent a small poisoned fraction from dominating the contrastive objective.

PRCL combines:
1) **Forensics signal** (label-free)
2) **Robust aggregation / bounded influence training**
3) **Negative-set sanitation**
4) Optional **quarantine + re-checking** (and optional lightweight purification)

### Probe-Consistency Forensics (PCF)
For each training sample `x`, PRCL computes a suspicion score `q(x) ∈ [0, 1]` based on representation stability under **probe transformations**.

**Standard SSL views**: your usual SimCLR/MoCo augmentations.  
**Probe views**: cheap transforms designed to stress shortcut triggers (e.g., mild occlusion, blur/noise, frequency attenuation, light color normalization).

Example PCF statistics (v1 chooses one simple primary statistic; more can be added):
- **Neighborhood overlap stability**: overlap of top-`k` nearest neighbors across probes
- **Probe alignment stability**: mean cosine similarity between embedding of `x` and embedding of `probe(x)`
- **Cross-view margin**: `(pos_sim − max_neg_sim)` measured under probes

Outputs:
- `q(x)` suspicion score per sample
- (optional) a binary decision via thresholding (used only for quarantine / reporting, not required for training)

**Design goals:**
- interpretable (“this sample behaves unusually under probes”),
- modular (plugs into SimCLR/MoCo),
- analyzable (bounded-influence / robust stats story).

### Robust contrastive optimization
PRCL reduces the influence of suspicious samples without requiring labels:
- **Downweight positives** from suspicious anchors (and/or suspicious pairs)
- **Cap per-sample gradient norms** (bounded influence)
- Optional **trim** top-`α` suspicious samples per batch/epoch (robust estimation trick)

### Poison-safe negatives
Contrastive learning uses large negative sets; poisoned negatives can warp the geometry.
PRCL reweights negatives to reduce poison influence:
- `w_neg(k) = (1 − q(x_k)) * r_ik`
where `r_ik` can additionally downweight near-duplicates / likely false negatives.

### Quarantine buffer (optional)
A reversible buffer for high-`q` samples:
- exclude from main training for `N` epochs,
- periodically re-evaluate (some samples may become “less suspicious” as representation improves),
- optionally run a lightweight purification step (e.g., distillation/unlearning-style) **only if needed**.


## What IntegritySuite provides
IntegritySuite is a focused integrity benchmark harness for SSL pretraining (not a general-purpose backdoor benchmark).

It provides:
- A unified config schema for: dataset / backbone / SSL baseline / attack / defense
- Standardized training & evaluation entry points
- Consistent reporting outputs (JSON/CSV) + plotting notebooks
- “Cards” to document attack/defense settings (threat model, knobs, failure modes)
- Sweeps and ablations with deterministic seeds

**Key deliverable:** “one command reproduces main tables” (after data download + environment setup).

---

## Quickstart
### 1) Create environment
```bash
conda create -n prcl python=3.10 -y
conda activate prcl
pip install -e .
````

### 2) Sanity run: clean SimCLR pretraining (CIFAR-10, ResNet-18)

```bash
python -m scripts.train \
  --config configs/ssl/simclr_cifar10_r18.yaml
```

### 3) Run PRCL defense (same setup)

```bash
python -m scripts.train \
  --config configs/prcl/prcl_simclr_cifar10_r18.yaml
```

### 4) Evaluate representation utility (linear probe + optional fine-tune)

```bash
python -m scripts.eval \
  --config configs/eval/linear_probe_cifar10.yaml \
  --ckpt runs/<RUN_ID>/checkpoints/last.ckpt
```

### 5) (Optional) Run IntegritySuite report aggregation

```bash
python -m integritysuite.report \
  --runs_dir runs/ \
  --out_dir reports/
```

> ⚠️ Attack runs are disabled by default. See [Ethics & responsible use](#ethics--responsible-use).


## Installation

### Requirements

* Python 3.10+
* PyTorch (GPU recommended)
* torchvision, numpy, scipy, scikit-learn
* (Optional) wandb / tensorboard for logging
* (Optional) faiss for kNN-heavy analysis (PCF variants can use CPU fallback)

### Install

```bash
pip install -e ".[dev]"
```

### Optional extras

```bash
pip install -e ".[wandb]"
pip install -e ".[faiss]"
```


## Repo layout

```
.
├── configs/                 # YAML configs (datasets, SSL, PRCL, eval, sweeps)
├── ssl/                     # SimCLR/MoCo implementations + encoders + heads
├── defenses/
│   └── prcl/                # PCF scoring, robust loss, negative sanitation, quarantine
├── attacks/                 # Thin wrappers / adapters around published implementations
├── eval/                    # Linear probe, fine-tune, ASR evaluation (defense-focused)
├── integritysuite/          # Runner + reporting + cards + schemas
├── scripts/                 # train/eval/sweep entrypoints
├── docs/                    # threat model, reproducibility, responsible release
└── runs/                    # outputs: logs, checkpoints, metrics, cards
```


## Configuration

All experiments are driven by YAML configs.

### Minimal config fields (conceptual)

* `dataset`: name, splits, transforms
* `model`: backbone, projection head
* `ssl`: simclr/moco, batch size, temperature, epochs
* `defense`: `none` or `prcl` + knobs
* `integritysuite`: logging, cards, metrics schema
* `seed`: full seed control
* `hardware`: device, amp, num_workers, determinism flags

### PRCL knobs (v1 defaults are conservative)

* `pcf.enabled`: true/false
* `pcf.k_probes`: 2 (start small), sweep later
* `pcf.stat`: `neighbor_overlap` (or `probe_alignment`, `margin`)
* `robust.mode`: `soft_weight` (utility-preserving default)
* `robust.trim_alpha`: 0.0 (off by default)
* `robust.grad_cap`: on (bounded influence)
* `negatives.reweight`: on
* `quarantine.enabled`: off by default


## Training

### Standard SSL (baseline)

```bash
python -m scripts.train --config configs/ssl/simclr_cifar10_r18.yaml
```

### PRCL

```bash
python -m scripts.train --config configs/prcl/prcl_simclr_cifar10_r18.yaml
```

### Sweeps (poison ratio / probes / ablations)

```bash
python -m scripts.sweep \
  --sweep_config configs/sweeps/prcl_kprobes_poisonratio.yaml
```


## Evaluation

IntegritySuite standardizes evaluation so results are comparable across runs.

### Utility metrics

* **Linear probe** top-1 accuracy
* **Fine-tune** accuracy (few-shot and full)

### Security metrics (defense evaluation)

* **ASR (Attack Success Rate)** on triggered inputs (when attacks are enabled and available)
* Clean accuracy under identical downstream settings

### Forensics metrics (PCF quality)

* detection ROC-AUC / PR-AUC (when ground-truth poisoned indices exist in controlled experiments)
* quarantine size vs. utility / ASR tradeoff curves

### Representation metrics (optional)

* kNN consistency under probes
* CKA / subspace similarity drift
* cluster separability stats


## Metrics & reporting

Each run writes:

* `metrics.json` (single source of truth)
* `cards/attack_card.md` (if enabled)
* `cards/defense_card.md`
* `config_resolved.yaml` (fully resolved config for exact reproduction)
* `checkpoints/` (last + best)

Reports:

* Aggregated CSV/JSON leaderboards
* Plotting notebook(s) under `integritysuite/plots/`


## Reproducibility

This repo is engineered for paper-grade reproducibility:

* Full seed control (torch/cuda/dataloader)
* Config resolution saved per run
* Deterministic flags where feasible
* Fixed evaluation scripts and schemas
* Single-command reproduction targets (main tables, ablations)

Recommended logging:

* software versions (`pip freeze`)
* GPU model / CUDA version
* runtime per epoch
* probe `K`, robust mode, thresholds, trimming alpha
* poison ratio and any *attack metadata* (stored only when explicitly enabled)


## Theory (partial guarantees)

PRCL aims for **clean, defensible partial theory** under a contamination model:

* `X ~ (1−ρ)P + ρQ` with small `ρ`

Target statements:

* Robust objective deviation: `| L̂_PRCL(θ) − L(θ;P) | ≤ O(ρ) + ε_n`
* Robust gradient deviation: `||∇L̂_PRCL(θ) − ∇L(θ;P)|| ≤ O(ρ) + ε'_n`

Proof sketch direction:

* PCF + robust weighting as bounded-influence estimator
* trimmed mean / Huber-style analysis + concentration for Monte Carlo probes

> The repo will include `docs/theory.md` that states assumptions explicitly and matches experiments to the theory narrative.


## Ethics & responsible use

This project focuses on **defense research and integrity evaluation**.

### Safe defaults

* Attack execution is **disabled by default**.
* The repository does **not** ship a “one-click attack toolkit.”
* Where attack evaluation is supported, it is done via **thin adapters** that rely on *external, published research implementations* and require explicit opt-in.

### Explicit opt-in required

To enable any attack-related evaluation in IntegritySuite:

1. Set an environment variable:

```bash
export INTEGRITYSUITE_ENABLE_ATTACKS=1
```

2. Provide an explicit CLI flag:

```bash
python -m scripts.train --config <...> --i_understand_attacks_are_for_research_only
```

If you are releasing results publicly:

* include the threat model,
* disclose safe defaults,
* avoid releasing sensitive operational details beyond what is required for reproducibility.


## Roadmap

### v0 (fast iteration)

* CIFAR-10 / STL-10 + ResNet-18
* SimCLR baseline + PRCL v1
* PCF statistic v1 + robust weighting v1
* Linear probe utility + basic reporting

### v1 (paper-ready)

* CIFAR-100 + TinyImageNet-200
* ResNet-50 and at least one additional backbone
* Strong ablations (PCF off, robust off, negatives off, K sweep, ρ sweep)
* IntegritySuite stable schemas + “main table reproduce” scripts

### v2 (If time permits)

* Optional ImageNet-100
* Optional small ViT experiments
* Optional adaptive attacker discussion experiment


### Acknowledgments

This repository integrates ideas from robust statistics, contrastive SSL, and integrity evaluation. Where attack adapters are used, they reference and respect upstream research implementations and licenses.

```


