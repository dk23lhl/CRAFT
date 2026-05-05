#!/usr/bin/env bash
# B5 — lambda_repr sensitivity sweep.
#
# 24 runs: 4 configs x 2 datasets x 3 seeds.
#   lambda_repr in {0.1, 0.5, 2.0, 5.0}
#   lambda=0.0 reuses B4_*_recon_only_s1.
#   lambda=1.0 reuses B4_*_full.
#
# Motivation: B4 on CUB shows recon_only (lambda=0) is 18pp higher than full (lambda=1).
#       Reviewers will ask "what if you just turn lambda down?" — need a full sensitivity curve.
#       Run both datasets: HW expected to be stable in lambda in [0.5, 2.0] (SimSiam helps),
#       CUB expected to decrease monotonically (SimSiam hurts on V=2).
#
# Naming: lambda_repr 0.1 -> lr01, 0.5 -> lr05, 2.0 -> lr20, 5.0 -> lr50.
#       Note that "lr" here abbreviates lambda_repr, not learning_rate.
#
# Overrides:
#   SEEDS="42" bash scripts/ablations/run_b5.sh        # smoke
#   DRY_RUN=1 bash scripts/ablations/run_b5.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${SEEDS:=42 43 44}"
lambdas=(0.1 0.5 2.0 5.0)
datasets=(hw cub)

n_seeds=$(echo "$SEEDS" | wc -w | tr -d ' ')
total=$(( ${#lambdas[@]} * ${#datasets[@]} * n_seeds ))

echo "B5: lambda_repr sensitivity"
echo "  lambdas : ${lambdas[*]} (lambda=0/1 reuse B4_recon_only/B4_full)"
echo "  datasets: ${datasets[*]}"
echo "  seeds   : $SEEDS"
echo "  total   : $total runs"

i=0
for L in "${lambdas[@]}"; do
  lr_tag=$(echo "$L" | tr -d '.')
  for dataset in "${datasets[@]}"; do
    base=$(${dataset}_args)
    for seed in $SEEDS; do
      i=$((i+1))
      exp="B5_${dataset}_lr${lr_tag}_s${seed}"
      echo "[$i/$total] $exp"
      # shellcheck disable=SC2086
      run_if_new "$exp" $base --lambda_repr "$L" --seed "$seed"
    done
  done
done

echo "B5 done. Aggregate: python aggregate_results.py --filter B5_ --out b5.csv"
