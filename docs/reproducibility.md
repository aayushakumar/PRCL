# Reproducibility

## Run Artifacts

Every experiment run creates a structured directory under `runs/` containing:

| Artifact | Purpose |
|----------|---------|
| `metadata.json` | Git commit hash, seed, CLI command, timestamp |
| `hardware.json` | CPU model, GPU info, CUDA version, memory |
| `env.txt` | Full `pip freeze` output |
| `metrics.json` | Per-epoch training metrics + eval results |
| `cards/run_card.json` | Structured experiment summary |
| `cards/run_card.md` | Human-readable experiment card |
| `checkpoints/best.ckpt` | Best model (by training loss) |
| `checkpoints/last.ckpt` | Final epoch model |
| `attack_metadata.json` | Attack config (only when attacks enabled) |

## Reproducing Results

### Quick smoke test
```bash
EPOCHS=5 SUBSET=5000 bash scripts/reproduce_main_tables.sh
```

### Full reproduction
```bash
export PRCL_ALLOW_ATTACKS=1
bash scripts/reproduce_main_tables.sh
```

### From a saved config
Each run saves its resolved config. To exactly replay:
```bash
python scripts/train.py --config-dir /path/to/run/  # uses saved config
```

## Seed Control

Full determinism is controlled via config:
```yaml
seed: 42
reproducibility:
  deterministic: true   # torch.use_deterministic_algorithms
  benchmark: false       # torch.backends.cudnn.benchmark
```

Seeds are set for: `torch`, `numpy`, `random`, and CUDA (all devices).

## Environment

Tested with:
- Python 3.10+
- PyTorch 2.0+
- Ubuntu 20.04+ / Debian-based systems
- NVIDIA GPUs (A100, V100, T4) and CPU-only

The `env.txt` file in each run captures the exact package versions used.
