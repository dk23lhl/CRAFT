#!/usr/bin/env bash
# B8 — C1/C2 violation experiment: transformer (CRAFT baseline) vs concat_mlp (violates C1+C2).
#
# 20 training runs: 2 fusion x 2 dataset x 5 seeds.
# Trained on complete data; Cell-Level missing-view evaluation is run separately via
# incomplete_eval.py after training.
#
# Overrides:
#   SEEDS="42 123" bash scripts/ablations/run_b8.sh       # only 2 seeds for smoke
#   DRY_RUN=1 bash scripts/ablations/run_b8.sh            # print commands only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${SEEDS:=42 43 44 45 46}"

datasets=(hw cub)
fusions=(transformer concat_mlp)

n_seeds=$(echo "$SEEDS" | wc -w | tr -d ' ')
total=$(( ${#fusions[@]} * ${#datasets[@]} * n_seeds ))

echo "B8: fusion_type ablation"
echo "  fusions : ${fusions[*]}"
echo "  datasets: ${datasets[*]}"
echo "  seeds   : $SEEDS"
echo "  total   : $total runs"

i=0
for fusion in "${fusions[@]}"; do
  for dataset in "${datasets[@]}"; do
    base=$(${dataset}_args)
    for seed in $SEEDS; do
      i=$((i+1))
      exp="B8_${dataset}_${fusion}_s${seed}"
      echo "[$i/$total] $exp"
      # shellcheck disable=SC2086
      run_if_new "$exp" $base --fusion_type "$fusion" --seed "$seed"
    done
  done
done

echo "B8 done. Next: run incomplete_eval.py on each ckpt for Cell-Level mr=0.3/0.5."
