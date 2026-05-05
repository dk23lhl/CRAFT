#!/usr/bin/env bash
# Reproduce paper Table 22 (Appx E.4): missing-view strategy comparison.
#
# 4 strategies × CUB × Protocol 4 × {0.1, 0.3, 0.5}:
#   - Attention masking (canonical CRAFT, strict C2)
#   - Learnable placeholder
#   - Zero imputation
#   - Mean imputation
#
# All 4 share the SAME complete-trained ckpt; only inference differs.
# Wraps run_a2.sh (training) + run_eval_cub_4strat.sh (eval with --mask_strategy override).
#
# Usage:
#   bash scripts/reproduce_strategy.sh
#   GPU=1 PARALLEL=4 bash scripts/reproduce_strategy.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SEEDS:=42 43 44}"
: "${GPU:=0}"
: "${PARALLEL:=4}"

echo "========================================"
echo "Reproduce Table 22 (Appx E.4): Missing-View Strategy Comparison"
echo "  seeds=$SEEDS  GPU=$GPU  PARALLEL=$PARALLEL"
echo "========================================"

# Phase 1: train CUB ckpts with each mask_strategy (4 ckpts × 3 seeds = 12)
GPU=$GPU PARALLEL=$PARALLEL SEEDS="$SEEDS" \
  bash "$SCRIPT_DIR/ablations/run_a2.sh"

# Phase 2: 4-protocol IMVC eval (no strategy override; uses each ckpt's own training strategy)
GPU=$GPU PARALLEL=$PARALLEL PROTOCOL=all NUM_TRIALS=5 \
  bash "$SCRIPT_DIR/ablations/run_eval_b8_a2.sh"

python aggregate_imvc_results.py --filter A2_ --protocol cell_level --out e4_strategy.csv
python summarize_csv.py --csv e4_strategy.csv --out e4_strategy_summary.md

echo "========================================"
echo "Done. Compare e4_strategy_summary.md with paper Table 22."
echo "========================================"
