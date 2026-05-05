#!/usr/bin/env bash
# A2 — Missing-view strategy ablation (fills CUB; HW already done).
#
# 12 runs: 4 strategies x CUB x 3 seeds.
#
# Training uses complete data throughout (no --missing_rate / --train_mask_prob).
# mask_strategy only takes effect at inference time when views are missing, so the protocol is:
#   1) Train 4 models (one per strategy) on complete data.
#   2) After training, evaluate each ckpt with incomplete_eval.py at Cell-Level mr=0.5.
#
# Note: under complete training, the learnable mask_token is never triggered and stays at
#       random init; this is the core question the paper answers — "do strategies still work
#       without MFT?".

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${SEEDS:=42 43 44}"

strategies=(learnable zero mean attention_mask)
datasets=(cub)

n_seeds=$(echo "$SEEDS" | wc -w | tr -d ' ')
total=$(( ${#strategies[@]} * ${#datasets[@]} * n_seeds ))

echo "A2: mask_strategy ablation (CUB only, HW already done)"
echo "  strategies: ${strategies[*]}"
echo "  datasets  : ${datasets[*]}"
echo "  seeds     : $SEEDS"
echo "  total     : $total runs"

i=0
for strategy in "${strategies[@]}"; do
  for dataset in "${datasets[@]}"; do
    base=$(${dataset}_args)
    for seed in $SEEDS; do
      i=$((i+1))
      exp="A2_${dataset}_${strategy}_s${seed}"
      echo "[$i/$total] $exp"
      # shellcheck disable=SC2086
      run_if_new "$exp" $base --mask_strategy "$strategy" --seed "$seed"
    done
  done
done

echo "A2 done. Next: for each ckpt, run"
echo "  python incomplete_eval.py --checkpoint <ckpt> --dataset cub \\"
echo "      --protocol cell_level --missing_rates 0.5 --num_trials 5 --seed 42"
