#!/usr/bin/env bash
# Reproduce paper Table 20 (Appx E.1): loss & training-stage component ablation.
#
# 6 configs × {HW, CUB} × 3 seeds = 36 training runs, evaluated under Protocol 4.
# Configs: Full / No Stage 2 / No Entropy / Recon Only / Repr Only / No Pretrain.
# Wraps the underlying ablation script run_b4.sh.
#
# Usage:
#   bash scripts/reproduce_component_ablation.sh
#   GPU=1 PARALLEL=4 bash scripts/reproduce_component_ablation.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SEEDS:=42 43 44}"
: "${GPU:=0}"
: "${PARALLEL:=4}"

echo "========================================"
echo "Reproduce Table 20 (Appx E.1): Loss & Stage Ablation"
echo "  seeds=$SEEDS  GPU=$GPU  PARALLEL=$PARALLEL"
echo "========================================"

# Phase 1: Train (run_b4.sh handles 6 configs × 2 datasets × 3 seeds)
GPU=$GPU PARALLEL=$PARALLEL SEEDS="$SEEDS" \
  bash "$SCRIPT_DIR/ablations/run_b4.sh"

# Phase 2: 4-protocol IMVC eval for all B4 ckpts
GPU=$GPU PARALLEL=$PARALLEL PROTOCOL=all NUM_TRIALS=5 \
  bash "$SCRIPT_DIR/ablations/run_eval_b4.sh"

# Phase 3: Aggregate
python aggregate_imvc_results.py --filter B4_ --protocol cell_level --out e1_component_ablation.csv
python summarize_csv.py --csv e1_component_ablation.csv --out e1_component_ablation_summary.md

echo "========================================"
echo "Done. Compare e1_component_ablation_summary.md with paper Table 20."
echo "========================================"
