#!/usr/bin/env bash
# ============================================================
# Poison-ratio sweep: evaluate PRCL across contamination levels.
#
# Runs undefended + PRCL-defended SimCLR at multiple poison ratios
# for a given attack family. Designed for Colab Pro or GPU cluster.
#
# Usage:
#   export PRCL_ALLOW_ATTACKS=1
#   bash scripts/sweep_poison_ratio.sh [ATTACK] [DATASET] [EPOCHS]
#
# Example:
#   bash scripts/sweep_poison_ratio.sh patch_backdoor cifar10 200
# ============================================================
set -euo pipefail

ATTACK="${1:-patch_backdoor}"
DATASET="${2:-cifar10}"
EPOCHS="${3:-200}"
SEED="${SEED:-42}"
RATIOS="${RATIOS:-0.005 0.01 0.03 0.05 0.10}"

echo "=== Poison-Ratio Sweep ==="
echo "  Attack:  $ATTACK"
echo "  Dataset: $DATASET"
echo "  Epochs:  $EPOCHS"
echo "  Ratios:  $RATIOS"
echo "  Seed:    $SEED"
echo ""

# Clean baseline (no attack)
echo "--- [1] Clean baseline (no attack, no defense) ---"
python scripts/train.py \
    dataset=$DATASET \
    ssl.epochs=$EPOCHS \
    seed=$SEED \
    attack=none \
    defense=none

# Clean + PRCL (verify no accuracy degradation)
echo "--- [2] Clean + PRCL defense (no attack) ---"
python scripts/train.py \
    dataset=$DATASET \
    ssl.epochs=$EPOCHS \
    seed=$SEED \
    attack=none \
    defense=prcl

# Sweep poison ratios: undefended + defended
for RATIO in $RATIOS; do
    echo "--- [3] Undefended, $ATTACK, ratio=$RATIO ---"
    python scripts/train.py \
        dataset=$DATASET \
        ssl.epochs=$EPOCHS \
        seed=$SEED \
        attack=$ATTACK \
        attack.poison_ratio=$RATIO \
        defense=none

    echo "--- [4] PRCL-defended, $ATTACK, ratio=$RATIO ---"
    python scripts/train.py \
        dataset=$DATASET \
        ssl.epochs=$EPOCHS \
        seed=$SEED \
        attack=$ATTACK \
        attack.poison_ratio=$RATIO \
        defense=prcl
done

echo ""
echo "=== Sweep complete. Generate report: python scripts/report.py --runs-dir ./runs ==="
