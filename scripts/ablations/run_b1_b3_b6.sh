#!/usr/bin/env bash
# B1 + B3 + B6 — Architecture and hyperparameter sensitivity.
#
# B1 layers       : fusion_n_layers in {1, 2, 3, 4}   (baseline=1)
# B3 embed dim    : embed_dim in {64, 128, 256, 512}  (HW baseline=256, CUB baseline=128)
# B6 cluster temp : cluster_temperature in {0.1, 0.2, 0.3, 0.5, 0.7} (baseline=0.5)
#
# Total 78 runs: (4+4+5) x 2 datasets x 3 seeds.
#
# Note: B3 d=256 on HW / d=128 on CUB are equivalent to baseline, retrained for seed consistency.
#       B6 tau=0.5 on both datasets is equivalent to baseline, same reason.
#       Baseline comparison values can reuse B4_{hw,cub}_full_s{42,123,456}.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${SEEDS:=42 43 44}"

datasets=(hw cub)
n_seeds=$(echo "$SEEDS" | wc -w | tr -d ' ')

layers=(1 2 3 4)
embed_dims=(64 128 256 512)
temps=(0.1 0.2 0.3 0.5 0.7)

total=$(( (${#layers[@]} + ${#embed_dims[@]} + ${#temps[@]}) * ${#datasets[@]} * n_seeds ))

echo "B1 + B3 + B6: architecture & hyperparameter sensitivity"
echo "  B1 layers: ${layers[*]}"
echo "  B3 dims  : ${embed_dims[*]}"
echo "  B6 temps : ${temps[*]}"
echo "  datasets : ${datasets[*]}"
echo "  seeds    : $SEEDS"
echo "  total    : $total runs"

i=0

echo ""
echo "--- B1: fusion_n_layers ---"
for L in "${layers[@]}"; do
  for dataset in "${datasets[@]}"; do
    base=$(${dataset}_args)
    for seed in $SEEDS; do
      i=$((i+1))
      exp="B1_${dataset}_L${L}_s${seed}"
      echo "[$i/$total] $exp"
      # shellcheck disable=SC2086
      run_if_new "$exp" $base --fusion_n_layers "$L" --seed "$seed"
    done
  done
done

echo ""
echo "--- B3: embed_dim ---"
for D in "${embed_dims[@]}"; do
  for dataset in "${datasets[@]}"; do
    base=$(${dataset}_args)
    for seed in $SEEDS; do
      i=$((i+1))
      exp="B3_${dataset}_d${D}_s${seed}"
      echo "[$i/$total] $exp"
      # shellcheck disable=SC2086
      run_if_new "$exp" $base --embed_dim "$D" --seed "$seed"
    done
  done
done

echo ""
echo "--- B6: cluster_temperature ---"
for T in "${temps[@]}"; do
  tau_tag=$(echo "$T" | tr -d '.')
  for dataset in "${datasets[@]}"; do
    base=$(${dataset}_args)
    for seed in $SEEDS; do
      i=$((i+1))
      exp="B6_${dataset}_tau${tau_tag}_s${seed}"
      echo "[$i/$total] $exp"
      # shellcheck disable=SC2086
      run_if_new "$exp" $base --cluster_temperature "$T" --seed "$seed"
    done
  done
done

echo "B1+B3+B6 done. Aggregate:"
echo "  python aggregate_results.py --filter B1_ --out b1.csv"
echo "  python aggregate_results.py --filter B3_ --out b3.csv"
echo "  python aggregate_results.py --filter B6_ --out b6.csv"
