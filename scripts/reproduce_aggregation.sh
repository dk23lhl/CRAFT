#!/usr/bin/env bash
# Reproduce paper Table 23 (Appx E.5): aggregation strategy comparison.
#
# 3 aggregation methods × {HW, CUB} × Protocol 4 × {0.1, 0.3, 0.5, 0.7}:
#   - CLS token (canonical CRAFT)
#   - Mean pooling
#   - Max pooling
#
# Wraps run_b7.sh (training) + run_eval_b2b5b7.sh (eval).
#
# Usage:
#   bash scripts/reproduce_aggregation.sh
#   GPU=1 PARALLEL=4 bash scripts/reproduce_aggregation.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SEEDS:=42 43 44}"
: "${GPU:=0}"
: "${PARALLEL:=4}"

echo "========================================"
echo "Reproduce Table 23 (Appx E.5): Aggregation Strategy"
echo "  seeds=$SEEDS  GPU=$GPU  PARALLEL=$PARALLEL"
echo "========================================"

GPU=$GPU PARALLEL=$PARALLEL SEEDS="$SEEDS" \
  bash "$SCRIPT_DIR/ablations/run_b7.sh"

GPU=$GPU PARALLEL=$PARALLEL PROTOCOL=all NUM_TRIALS=5 \
  bash "$SCRIPT_DIR/ablations/run_eval_b2b5b7.sh"

python aggregate_imvc_results.py --filter B7_ --protocol cell_level --out e5_aggregation.csv
python summarize_csv.py --csv e5_aggregation.csv --out e5_aggregation_summary.md

echo "========================================"
echo "Done. Compare e5_aggregation_summary.md with paper Table 23."
echo "========================================"
