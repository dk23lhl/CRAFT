#!/usr/bin/env bash
# Reproduce paper Table 21 (Appx E.2): CRAFT-Core (theory-only minimum).
#
# CRAFT-Core = C1 + C2 only:
#   - Stage 1 reconstruction loss  ✓
#   - Stage 2 K-Means cluster head init  ✓
#   - Attention masking at inference  ✓
#   - REMOVED: SimSiam consistency, entropy reg, KL reg, MFT, Stage 2 cluster fine-tuning
#
# {HW, CUB} × 3 seeds × 4 protocols × 4 missing rates.
# Wraps run_core.sh + run_eval_core.sh.
#
# Usage:
#   bash scripts/reproduce_craft_core.sh
#   GPU=1 PARALLEL=4 bash scripts/reproduce_craft_core.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SEEDS:=42 43 44}"
: "${GPU:=0}"
: "${PARALLEL:=4}"

echo "========================================"
echo "Reproduce Table 21 (Appx E.2): CRAFT-Core"
echo "  seeds=$SEEDS  GPU=$GPU  PARALLEL=$PARALLEL"
echo "========================================"

GPU=$GPU PARALLEL=$PARALLEL SEEDS="$SEEDS" \
  bash "$SCRIPT_DIR/ablations/run_core.sh"

GPU=$GPU PARALLEL=$PARALLEL PROTOCOL=all NUM_TRIALS=5 \
  bash "$SCRIPT_DIR/ablations/run_eval_core.sh"

python aggregate_imvc_results.py --filter Core_ --out e2_craft_core.csv
python summarize_csv.py --csv e2_craft_core.csv --out e2_craft_core_summary.md

echo "========================================"
echo "Done. Compare e2_craft_core_summary.md with paper Table 21."
echo "========================================"
