#!/usr/bin/env bash
# B7 — Aggregation strategy ablation.
#
# 12 runs: 2 configs x 2 datasets x 3 seeds.
#   aggregation in {mean, max}
#   aggregation=cls reuses B1_*_L1 / B3_hw_d256 / B3_cub_d128 baseline.
#
# Implementation note:
#   - CLS token is still prepended (architecture unchanged); only "which part of the output to take" changes.
#   - mean/max only aggregate over observed view tokens (view_mask excludes missing ones).
#   - See the aggregation branch in models/fusion.py:ViewFusionTransformer.forward.
#
# Motivation: all three aggregations satisfy C1 (permutation-equivariant) and C2 (mask-aware).
#       Expected difference < 2pp, further supporting the "C1/C2 is load-bearing, specific
#       aggregation is secondary" argument. Complements B8 (transformer vs concat_mlp): B8 shows
#       that breaking C1/C2 degrades significantly; B7 shows that within C1/C2, implementation
#       differences are minimal.
#
# Overrides:
#   SEEDS="42" bash scripts/ablations/run_b7.sh        # smoke
#   DRY_RUN=1 bash scripts/ablations/run_b7.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${SEEDS:=42 43 44}"
aggs=(mean max)
datasets=(hw cub)

n_seeds=$(echo "$SEEDS" | wc -w | tr -d ' ')
total=$(( ${#aggs[@]} * ${#datasets[@]} * n_seeds ))

echo "B7: aggregation strategy ablation"
echo "  aggregations: ${aggs[*]} (cls reuses B1_*_L1 baseline)"
echo "  datasets    : ${datasets[*]}"
echo "  seeds       : $SEEDS"
echo "  total       : $total runs"

i=0
for A in "${aggs[@]}"; do
  for dataset in "${datasets[@]}"; do
    base=$(${dataset}_args)
    for seed in $SEEDS; do
      i=$((i+1))
      exp="B7_${dataset}_${A}_s${seed}"
      echo "[$i/$total] $exp"
      # shellcheck disable=SC2086
      run_if_new "$exp" $base --aggregation "$A" --seed "$seed"
    done
  done
done

echo "B7 done. Aggregate: python aggregate_results.py --filter B7_ --out b7.csv"
