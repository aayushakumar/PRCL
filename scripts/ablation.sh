#!/usr/bin/env bash
# ============================================================
# PRCL Ablation Matrix
#
# Tests each PRCL component in isolation and combination to
# measure individual contributions. Matches PRD Phase 5 Step 5.4.
#
# Usage:
#   export PRCL_ALLOW_ATTACKS=1
#   bash scripts/ablation.sh [DATASET] [EPOCHS] [POISON_RATIO]
# ============================================================
set -euo pipefail

DATASET="${1:-cifar10}"
EPOCHS="${2:-200}"
RATIO="${3:-0.01}"
SEED="${SEED:-42}"
ATTACK="${ATTACK:-patch_backdoor}"

echo "=== PRCL Ablation Matrix ==="
echo "  Dataset: $DATASET  |  Epochs: $EPOCHS  |  Ratio: $RATIO  |  Attack: $ATTACK"
echo ""

configs=(
    # label:defense_overrides
    "no_defense:defense=none"
    "pcf_only:defense=prcl defense.pcf.enabled=true defense.robust.mode=none defense.negatives.enabled=false"
    "pcf+pos_reweight:defense=prcl defense.pcf.enabled=true defense.robust.mode=soft_weight defense.negatives.enabled=false"
    "pcf+neg_sanitize:defense=prcl defense.pcf.enabled=true defense.robust.mode=none defense.negatives.enabled=true"
    "prcl_no_gradcap:defense=prcl defense.pcf.enabled=true defense.robust.mode=soft_weight defense.negatives.enabled=true defense.robust.grad_cap_enabled=false"
    "prcl_full:defense=prcl"
)

for entry in "${configs[@]}"; do
    label="${entry%%:*}"
    overrides="${entry#*:}"
    echo "--- [$label] ---"
    # shellcheck disable=SC2086
    python scripts/train.py \
        dataset=$DATASET \
        ssl.epochs=$EPOCHS \
        seed=$SEED \
        attack=$ATTACK \
        attack.poison_ratio=$RATIO \
        $overrides
    echo ""
done

echo "=== Ablation complete. Generate report: python scripts/report.py --runs-dir ./runs ==="
