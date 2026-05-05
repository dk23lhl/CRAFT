#!/usr/bin/env bash
# B2 — fusion_n_heads ablation.
#
# 18 runs: 3 configs x 2 datasets x 3 seeds.
#   heads in {1, 2, 8}
#   heads=4 reuses B1_*_L1 / B3_hw_d256 / B3_cub_d128 baseline.
#
# d must be divisible by heads:
#   HW d=256: 1/2/4/8 all OK.
#   CUB d=128: 1/2/4/8 all OK.
#
# Overrides:
#   SEEDS="42" bash scripts/ablations/run_b2.sh        # smoke
#   DRY_RUN=1 bash scripts/ablations/run_b2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${SEEDS:=42 43 44}"
heads=(1 2 8)
datasets=(hw cub)

n_seeds=$(echo "$SEEDS" | wc -w | tr -d ' ')
total=$(( ${#heads[@]} * ${#datasets[@]} * n_seeds ))

echo "B2: fusion_n_heads ablation"
echo "  heads   : ${heads[*]} (heads=4 reuses B1_*_L1 baseline)"
echo "  datasets: ${datasets[*]}"
echo "  seeds   : $SEEDS"
echo "  total   : $total runs"

i=0
for H in "${heads[@]}"; do
  for dataset in "${datasets[@]}"; do
    base=$(${dataset}_args)
    for seed in $SEEDS; do
      i=$((i+1))
      exp="B2_${dataset}_h${H}_s${seed}"
      echo "[$i/$total] $exp"
      # shellcheck disable=SC2086
      run_if_new "$exp" $base --fusion_n_heads "$H" --seed "$seed"
    done
  done
done

echo "B2 done. Aggregate: python aggregate_results.py --filter B2_ --out b2.csv"
