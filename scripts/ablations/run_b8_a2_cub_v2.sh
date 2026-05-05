#!/usr/bin/env bash
# B8v2 + A2v2 — CUB rerun with new optimal baseline (d64 + lambda_repr=0).
#
# Motivation: C1 grid search found the CUB optimal config:
#   --embed_dim 64 --lambda_repr 0.0  ->  ACC 0.8133
#   old baseline (d128 + lr=1.0)      ->  ACC 0.6178   (+19.6pp gap)
# The old B8/A2 comparisons were done on the wrong baseline and were unfair; need to redo
# under the new baseline.
#
# Total 22 runs:
#   B8v2: 2 fusion (transformer, concat_mlp) x 5 seeds (42-46) = 10
#   A2v2: 4 strategy (learnable, zero, mean, attention_mask) x 3 seeds (42-44) = 12
#
# B8v2_/A2v2_ naming does not conflict with B8_/A2_ (bash glob `B8_*` does not match `B8v2_*`).
#
# After training, use run_eval_b8_a2.sh (already extended to glob B8v2_/A2v2_) for IMVC eval.
#
# Overrides:
#   B8_SEEDS="42" A2_SEEDS="42" bash scripts/ablations/run_b8_a2_cub_v2.sh   # smoke
#   DRY_RUN=1 bash scripts/ablations/run_b8_a2_cub_v2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

# CUB v2 baseline = cub_args plus --embed_dim 64 --lambda_repr 0.0.
# argparse store-action: later value overrides earlier (cub_args' --embed_dim 128 -> 64).
cub_v2_args() {
  echo "$(cub_args) --embed_dim 64 --lambda_repr 0.0"
}

: "${B8_SEEDS:=42 43 44 45 46}"
b8_fusions=(transformer concat_mlp)
n_b8=$(echo "$B8_SEEDS" | wc -w | tr -d ' ')
total_b8=$(( ${#b8_fusions[@]} * n_b8 ))

echo "B8v2: fusion_type on CUB (new baseline d64+lr=0)"
echo "  fusions: ${b8_fusions[*]}"
echo "  seeds  : $B8_SEEDS"
echo "  total  : $total_b8 runs"

i=0
for fusion in "${b8_fusions[@]}"; do
  base=$(cub_v2_args)
  for seed in $B8_SEEDS; do
    i=$((i+1))
    exp="B8v2_cub_${fusion}_s${seed}"
    echo "[$i/$total_b8] $exp"
    # shellcheck disable=SC2086
    run_if_new "$exp" $base --fusion_type "$fusion" --seed "$seed"
  done
done

: "${A2_SEEDS:=42 43 44}"
a2_strategies=(learnable zero mean attention_mask)
n_a2=$(echo "$A2_SEEDS" | wc -w | tr -d ' ')
total_a2=$(( ${#a2_strategies[@]} * n_a2 ))

echo "A2v2: mask_strategy on CUB (new baseline d64+lr=0)"
echo "  strategies: ${a2_strategies[*]}"
echo "  seeds     : $A2_SEEDS"
echo "  total     : $total_a2 runs"

i=0
for strategy in "${a2_strategies[@]}"; do
  base=$(cub_v2_args)
  for seed in $A2_SEEDS; do
    i=$((i+1))
    exp="A2v2_cub_${strategy}_s${seed}"
    echo "[$i/$total_a2] $exp"
    # shellcheck disable=SC2086
    run_if_new "$exp" $base --mask_strategy "$strategy" --seed "$seed"
  done
done

echo "v2 training done ($((total_b8 + total_a2)) total)."
echo "Next:"
echo "  bash scripts/ablations/run_eval_b8_a2.sh   # IMVC eval (includes v2)"
echo "  python aggregate_results.py      --filter B8v2_  --out b8v2.csv"
echo "  python aggregate_results.py      --filter A2v2_  --out a2v2.csv"
echo "  python aggregate_imvc_results.py --filter B8v2_,A2v2_ --out imvc_v2.csv"
