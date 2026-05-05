#!/usr/bin/env bash
# Reproduce paper Table 14 (Appx D.6): rate-matched control on HandWritten.
#
# Per-rate trained CRAFT vs complete-trained CRAFT under matched (Protocol, r) cells.
# Reports both Protocol 1 (sample-level) and Protocol 4 (cell-level) on HW (V=6).
#
# 2 datasets × 2 protocols × 4 rates × 5 seeds:
#   - HW + Protocol 1 (completer)  — train at r ∈ {0.1, 0.3, 0.5, 0.7}
#   - HW + Protocol 4 (cell_level) — train at r ∈ {0.1, 0.3, 0.5, 0.7}
#
# Each per-rate ckpt is evaluated only at its matched (P, r) cell (see
# run_eval_b10.sh which auto-routes by ckpt name prefix B10p1_/B10p4_).
#
# Usage:
#   bash scripts/reproduce_rate_matched_control.sh
#   GPU=1 PARALLEL=4 bash scripts/reproduce_rate_matched_control.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SEEDS:=42 43 44 45 46}"
: "${RATES:=0.1 0.3 0.5 0.7}"
: "${GPU:=0}"
: "${PARALLEL:=4}"

echo "========================================"
echo "Reproduce Table 14 (Appx D.6): Rate-Matched Control on HW"
echo "  seeds=$SEEDS  rates=$RATES  GPU=$GPU  PARALLEL=$PARALLEL"
echo "========================================"

# Train Protocol 1 (completer) per-rate variants → B10p1_hw_*
echo ""
echo "=== Train Protocol 1 (sample-level) per-rate ==="
GPU=$GPU PARALLEL=$PARALLEL SEEDS="$SEEDS" RATES="$RATES" \
  DATASETS="hw" TRAIN_MASK_PROTOCOL=completer \
  bash "$SCRIPT_DIR/ablations/run_b10_train_incomplete.sh"

# Train Protocol 4 (cell_level) per-rate variants → B10p4_hw_*
echo ""
echo "=== Train Protocol 4 (cell-level) per-rate ==="
GPU=$GPU PARALLEL=$PARALLEL SEEDS="$SEEDS" RATES="$RATES" \
  DATASETS="hw" TRAIN_MASK_PROTOCOL=cell_level \
  bash "$SCRIPT_DIR/ablations/run_b10_train_incomplete.sh"

# Eval at matched (P, r) cell only
echo ""
echo "=== Eval matched cells only ==="
GPU=$GPU PARALLEL=$PARALLEL RATES="$RATES" PREFIXES="B10p1_hw_ B10p4_hw_" \
  bash "$SCRIPT_DIR/ablations/run_eval_b10.sh"

# Aggregate
python aggregate_imvc_results.py --filter "B10p1_hw_,B10p4_hw_" --out d6_rate_matched.csv
python summarize_csv.py --csv d6_rate_matched.csv --out d6_rate_matched_summary.md

echo "========================================"
echo "Done. Compare d6_rate_matched_summary.md with paper Table 14."
echo "  Note: paper Table 14 reports CUB Protocol 4 r∈{0.3, 0.5}; this script"
echo "  reproduces the HW counterpart used in §6.2's architectural-stability claim."
echo "========================================"
