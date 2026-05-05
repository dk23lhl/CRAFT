#!/usr/bin/env bash
# C1 — CUB optimal-config search (embed_dim x lambda_repr 2D grid).
#
# Motivation: B3 / B4 show that the current CUB baseline (d128 + full loss) is not locally optimal:
#   - B3: d64 (0.7350) > d128 (0.6178)  +11.7pp
#   - B4: recon_only (lr=0)  (0.7956) > full (0.6178)  +17.8pp
# Can these two directions be stacked? Can we go even smaller (d32)? Is an intermediate value
# (lr=0.5) more stable?
#
# Grid:
#   embed_dim    in {32, 64, 128}
#   lambda_repr  in {0.0, 0.5, 1.0}
#   seeds        = 42 43 44
# Total 3 x 3 x 3 = 27 runs.
#
# Naming: C1_cub_d<D>_lr<L_TAG>_s<seed>
#   L_TAG: 0.0 -> 00, 0.5 -> 05, 1.0 -> 10
#
# Note: the following 3 combinations are already done in B3/B4 (values should be close, useful as sanity check):
#   C1_cub_d128_lr10  ==  B4_cub_full           (0.6178 +/- 0.0160)
#   C1_cub_d64_lr10   ==  B3_cub_d64            (0.7350 +/- 0.0487)
#   C1_cub_d128_lr00  ==  B4_cub_recon_only_s1  (0.7956 +/- 0.0192)
#
# Overrides:
#   SEEDS="42" bash scripts/ablations/run_cub_search.sh   # smoke
#   DRY_RUN=1 bash scripts/ablations/run_cub_search.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${SEEDS:=42 43 44}"

embed_dims=(32 64 128)
lambda_reprs=(0.0 0.5 1.0)

n_seeds=$(echo "$SEEDS" | wc -w | tr -d ' ')
total=$(( ${#embed_dims[@]} * ${#lambda_reprs[@]} * n_seeds ))

echo "C1: CUB optimal config search"
echo "  embed_dims   : ${embed_dims[*]}"
echo "  lambda_reprs : ${lambda_reprs[*]}"
echo "  seeds        : $SEEDS"
echo "  total        : $total runs"

i=0
for d in "${embed_dims[@]}"; do
  for lr in "${lambda_reprs[@]}"; do
    lr_tag=$(echo "$lr" | tr -d '.')
    base=$(cub_args)
    for seed in $SEEDS; do
      i=$((i+1))
      exp="C1_cub_d${d}_lr${lr_tag}_s${seed}"
      echo "[$i/$total] $exp"
      # shellcheck disable=SC2086
      run_if_new "$exp" $base \
        --embed_dim "$d" \
        --lambda_repr "$lr" \
        --seed "$seed"
    done
  done
done

echo "C1 done. Aggregate with:"
echo "  python aggregate_results.py --filter C1_ --out c1.csv"
echo "  python summarize_csv.py --csv c1.csv --out c1_summary.md"
