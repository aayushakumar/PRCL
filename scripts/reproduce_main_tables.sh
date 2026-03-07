#!/usr/bin/env bash
# ============================================================
# Reproduce all main paper tables from scratch.
#
# Runs the full experiment matrix and generates CSV/LaTeX tables.
# This is the "one command" reproducibility script required by the PRD.
#
# Usage:
#   export PRCL_ALLOW_ATTACKS=1
#   bash scripts/reproduce_main_tables.sh
#
# For a quick smoke test (5 epochs, subset):
#   EPOCHS=5 SUBSET=5000 bash scripts/reproduce_main_tables.sh
# ============================================================
set -euo pipefail

EPOCHS="${EPOCHS:-200}"
SUBSET="${SUBSET:-}"
SEEDS="${SEEDS:-42 123 456}"

SUBSET_FLAG=""
if [ -n "$SUBSET" ]; then
    SUBSET_FLAG="dataset.subset_size=$SUBSET"
fi

echo "=== Reproducing Main Paper Tables ==="
echo "  Epochs:  $EPOCHS"
echo "  Seeds:   $SEEDS"
[ -n "$SUBSET" ] && echo "  Subset:  $SUBSET"
echo ""

# --- Table 1: Main results ---
for SEED in $SEEDS; do
    export SEED

    # CIFAR-10 experiments
    echo "--- CIFAR-10 seed=$SEED ---"
    bash scripts/sweep_poison_ratio.sh patch_backdoor cifar10 "$EPOCHS"

    # Blend attack (second family)
    for RATIO in 0.005 0.01 0.05; do
        echo "--- Blend attack, ratio=$RATIO, seed=$SEED ---"
        python scripts/train.py \
            dataset=cifar10 ssl.epochs="$EPOCHS" seed="$SEED" \
            attack=blend_backdoor attack.poison_ratio="$RATIO" \
            defense=none $SUBSET_FLAG

        python scripts/train.py \
            dataset=cifar10 ssl.epochs="$EPOCHS" seed="$SEED" \
            attack=blend_backdoor attack.poison_ratio="$RATIO" \
            defense=prcl $SUBSET_FLAG
    done
done

# --- Table 2: Ablation (single seed for brevity) ---
echo "=== Ablation matrix ==="
SEED="${SEED:-42}"
bash scripts/ablation.sh cifar10 "$EPOCHS" 0.01

# --- Generate tables ---
echo "=== Generating tables ==="
python scripts/report.py --runs-dir ./runs --output-dir ./tables

echo ""
echo "=== Done. Tables saved to ./tables/ ==="
