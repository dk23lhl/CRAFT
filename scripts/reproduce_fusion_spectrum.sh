#!/usr/bin/env bash
# Reproduce paper Table 19 (Appx D.11): fusion architecture spectrum on HandWritten.
#
# 4 fusion variants × HW × 4 protocols × 4 missing rates × 5 seeds:
#   - Transformer + attention masking (canonical CRAFT, strict C2)
#   - SetMLP-Light  (h'=192, capacity-matched to Transformer)
#   - SetMLP-Match  (h'=560, parameter-matched to concat fusion)
#   - Concat fusion (zero-fill, soft C2)
#
# Wraps run_b8.sh (Transformer + Concat fusion) + run_b9_setmlp.sh (SetMLP variants).
#
# Usage:
#   bash scripts/reproduce_fusion_spectrum.sh
#   GPU=1 PARALLEL=4 bash scripts/reproduce_fusion_spectrum.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SEEDS:=42 43 44 45 46}"
: "${GPU:=0}"
: "${PARALLEL:=4}"

echo "========================================"
echo "Reproduce Table 19 (Appx D.11): Fusion Architecture Spectrum"
echo "  seeds=$SEEDS  GPU=$GPU  PARALLEL=$PARALLEL"
echo "========================================"

# Transformer (canonical) + Concat fusion (HW)
GPU=$GPU PARALLEL=$PARALLEL SEEDS="$SEEDS" \
  bash "$SCRIPT_DIR/ablations/run_b8.sh"

# SetMLP-Light (h'=192) + SetMLP-Match (h'=560) on HW
GPU=$GPU PARALLEL=$PARALLEL SEEDS="$SEEDS" DATASETS="hw" \
  SETMLP_HIDDEN_HW=192 \
  bash "$SCRIPT_DIR/ablations/run_b9_setmlp.sh"
GPU=$GPU PARALLEL=$PARALLEL SEEDS="$SEEDS" DATASETS="hw" \
  SETMLP_HIDDEN_HW=560 \
  bash "$SCRIPT_DIR/ablations/run_b9_setmlp.sh"

# 4-protocol eval for all
GPU=$GPU PARALLEL=$PARALLEL PROTOCOL=all NUM_TRIALS=5 \
  bash "$SCRIPT_DIR/ablations/run_eval_b8_a2.sh"
GPU=$GPU PARALLEL=$PARALLEL PROTOCOL=all NUM_TRIALS=5 PREFIXES="B9_hw_" \
  bash "$SCRIPT_DIR/ablations/run_eval_b9_setmlp.sh"

python aggregate_imvc_results.py --filter "B8_hw_,B9_hw_" --out d11_fusion_spectrum.csv
python summarize_csv.py --csv d11_fusion_spectrum.csv --out d11_fusion_spectrum_summary.md

echo "========================================"
echo "Done. Compare d11_fusion_spectrum_summary.md with paper Table 19."
echo "========================================"
