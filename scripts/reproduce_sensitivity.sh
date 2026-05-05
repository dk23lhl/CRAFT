#!/usr/bin/env bash
# Reproduce paper Figure 4 (Appx E.3): hyperparameter sensitivity sweeps.
#
# 4 panels at Protocol 4, r=0.5, 3 seeds:
#   (a) Transformer depth     L  ∈ {1, 2, 3, 4}                  via run_b1_b3_b6.sh (B1)
#   (b) Embedding dim         d  ∈ {64, 128, 256, 512}           via run_b1_b3_b6.sh (B3)
#   (c) Repr loss weight      λ_repr ∈ {0.0, 0.5, 1.0} (CUB only) via run_cub_search.sh (C1)
#   (d) Cluster temperature   τ  ∈ {0.1, 0.2, 0.3, 0.5, 0.7}     via run_b1_b3_b6.sh (B6)
#
# Note: panel (c) covers 3 of 6 paper-reported λ_repr points (CUB only).
# Other points are derivable from existing B4 ablation (recon_only_s1 ≡ λ_repr=0).
#
# Usage:
#   bash scripts/reproduce_sensitivity.sh
#   GPU=1 PARALLEL=4 bash scripts/reproduce_sensitivity.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SEEDS:=42 43 44}"
: "${GPU:=0}"
: "${PARALLEL:=4}"

echo "========================================"
echo "Reproduce Figure 4 (Appx E.3): Hyperparameter Sensitivity"
echo "  seeds=$SEEDS  GPU=$GPU  PARALLEL=$PARALLEL"
echo "========================================"

# Panels (a) L, (b) d, (d) τ — both HW and CUB
GPU=$GPU PARALLEL=$PARALLEL SEEDS="$SEEDS" \
  bash "$SCRIPT_DIR/ablations/run_b1_b3_b6.sh"

# Panel (c) λ_repr — CUB 2D grid (embed_dim × λ_repr)
GPU=$GPU PARALLEL=$PARALLEL SEEDS="$SEEDS" \
  bash "$SCRIPT_DIR/ablations/run_cub_search.sh"

# IMVC eval at P4 r=0.5 only (sensitivity figure operating point)
GPU=$GPU PARALLEL=$PARALLEL PROTOCOL=cell_level MISSING_RATES="0.5" NUM_TRIALS=5 \
  bash "$SCRIPT_DIR/ablations/run_eval_b2b5b7.sh"   # B2/B5/B7 share the eval glob, also matches B1/B3/B6 — see script

python aggregate_imvc_results.py --filter "B1_,B3_,B6_,C1_" --protocol cell_level --out e3_sensitivity.csv
python summarize_csv.py --csv e3_sensitivity.csv --out e3_sensitivity_summary.md

echo "========================================"
echo "Done. Compare e3_sensitivity_summary.md with paper Figure 4."
echo "========================================"
