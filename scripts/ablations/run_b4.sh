#!/usr/bin/env bash
# B4 — Loss component ablation.
#
# 36 runs: 6 configs x 2 datasets x 3 seeds.
#
# Configs:
#   full            : baseline (Stage 1 recon+repr, Stage 2 cluster+entropy+recon+repr)
#   recon_only_s1   : disable repr (--lambda_repr 0 active throughout, see note below)
#   repr_only_s1    : disable recon (--lambda_recon 0 active throughout)
#   no_pretrain     : skip Stage 1 (--stage1_epochs 0, single-stage training)
#   no_stage2       : only Stage 1 + K-Means (--skip_stage2)
#   no_entropy_s2   : disable Stage 2 entropy regularization (--lambda_entropy 0)
#
# Note: main_mvc.py only restores lambda_cluster / lambda_entropy in Stage 2, not lambda_repr /
# lambda_recon. So recon_only_s1 / repr_only_s1 actually mean "loss disabled throughout", not
# "disabled in Stage 1 only". This semantics should be made explicit in the paper's ablation
# description to avoid misreading.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${SEEDS:=42 43 44}"

configs_order=(full recon_only_s1 repr_only_s1 no_pretrain no_stage2 no_entropy_s2)
declare -A configs_extra=(
  [full]=""
  [recon_only_s1]="--lambda_repr 0"
  [repr_only_s1]="--lambda_recon 0"
  [no_pretrain]="--stage1_epochs 0"
  [no_stage2]="--skip_stage2"
  [no_entropy_s2]="--lambda_entropy 0"
)

datasets=(hw cub)
n_seeds=$(echo "$SEEDS" | wc -w | tr -d ' ')
total=$(( ${#configs_order[@]} * ${#datasets[@]} * n_seeds ))

echo "B4: loss component ablation"
echo "  configs : ${configs_order[*]}"
echo "  datasets: ${datasets[*]}"
echo "  seeds   : $SEEDS"
echo "  total   : $total runs"

i=0
for cfg in "${configs_order[@]}"; do
  extra="${configs_extra[$cfg]}"
  for dataset in "${datasets[@]}"; do
    base=$(${dataset}_args)
    for seed in $SEEDS; do
      i=$((i+1))
      exp="B4_${dataset}_${cfg}_s${seed}"
      echo "[$i/$total] $exp"
      # shellcheck disable=SC2086
      run_if_new "$exp" $base $extra --seed "$seed"
    done
  done
done

echo "B4 done. Aggregate: python aggregate_results.py --filter B4_ --out b4.csv"
