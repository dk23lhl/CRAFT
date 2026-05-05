#!/usr/bin/env bash
# C3 — HW + CUB best-config 5-seed reproduction.
#
# Run the optimal config found by ablation on 5 seeds (42-46) for robustness + paper's final number.
# HW and CUB submitted in alternation, sharing the GPU (default GPU 0, configurable).
#
# Best configs (from ablation, two changes stacked):
#   HW :  hw_args() + --lambda_repr 0.5 --cluster_temperature 0.1
#         B5_hw_lr05    IMVC cell_level mr=0.5: 0.7721 vs default 0.7442 (+2.8pp)
#         B6_hw_tau01   training ACC          : 0.9663 vs default 0.9533 (+1.3pp)
#         The two changes are orthogonal (one adjusts loss weight, the other cluster-head
#         temperature); stacking expected to be effective.
#   CUB:  cub_args() + --lambda_repr 0.1
#         B5_cub_lr01   IMVC cell_level mr=0.1: 0.8182 vs default 0.7686 (+5.0pp)
#
# Total 10 runs: 5 seeds x 2 datasets.
#
# Overrides (recommended usage):
#   # Override HW best config (e.g. set lambda_repr 0.5)
#   HW_OVERRIDE="--lambda_repr 0.5" bash scripts/ablations/run_best_5seeds.sh
#
#   # No HW override (use default hw_args)
#   HW_OVERRIDE="" bash scripts/ablations/run_best_5seeds.sh
#
#   # Run on GPU 1 with 4 parallel
#   GPU=1 PARALLEL=4 bash scripts/ablations/run_best_5seeds.sh
#
#   # smoke
#   SEEDS="42" PARALLEL=1 bash scripts/ablations/run_best_5seeds.sh
#
#   DRY_RUN=1 bash scripts/ablations/run_best_5seeds.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${SEEDS:=42 43 44 45 46}"
: "${PARALLEL:=4}"
: "${GPU:=0}"
: "${HW_OVERRIDE:=--lambda_repr 0.5 --cluster_temperature 0.1}"
: "${CUB_OVERRIDE:=--lambda_repr 0.1}"

n_seeds=$(echo "$SEEDS" | wc -w | tr -d ' ')
total=$(( 2 * n_seeds ))

echo "C3 best-config 5-seed reproduction"
echo "  HW  override: ${HW_OVERRIDE:-(none, use default hw_args)}"
echo "  CUB override: ${CUB_OVERRIDE:-(none, use default cub_args)}"
echo "  seeds       : $SEEDS"
echo "  GPU         : $GPU"
echo "  parallel    : $PARALLEL"
echo "  total runs  : $total"

run_one() {
  local exp="$1"; shift
  local outdir="$OUTPUT_ROOT/$exp"
  local logfile="$LOG_DIR/$exp.log"
  if [[ -f "$outdir/results.yaml" ]]; then
    echo "    [skip] $exp (results exists)"
    return 0
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "    [dry gpu=$GPU] $exp"
    return 0
  fi
  echo "    [run gpu=$GPU] $exp -> $logfile"
  CUDA_VISIBLE_DEVICES="$GPU" python main_mvc.py "$@" \
    --exp_name "$exp" --output_dir "$(dirname "$OUTPUT_ROOT")" \
    > "$logfile" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "    [FAIL] $exp (exit $rc); see $logfile"
  fi
}

i=0
n_running=0
hw_base=$(hw_args)
cub_base=$(cub_args)

# Alternate HW/CUB submissions (same seed) so both datasets run in parallel.
for seed in $SEEDS; do
  for combo in hw cub; do
    i=$((i+1))
    if [[ "$combo" == "hw" ]]; then
      exp="C3_best_hw_s${seed}"
      base="$hw_base"
      override="$HW_OVERRIDE"
    else
      exp="C3_best_cub_s${seed}"
      base="$cub_base"
      override="$CUB_OVERRIDE"
    fi

    while (( n_running >= PARALLEL )); do
      wait -n
      n_running=$((n_running-1))
    done

    # shellcheck disable=SC2086
    run_one "$exp" $base $override --seed "$seed" &
    n_running=$((n_running+1))
    echo "[$i/$total] launched $exp (in_flight=$n_running)"
  done
done

echo "All $total launched. Waiting for last $n_running in flight..."
wait
echo "Training done."
echo ""
echo "Next: bash scripts/ablations/run_eval_best_5seeds.sh   # IMVC 4-protocol eval"
echo ""
echo "Aggregate training:"
echo "  python aggregate_results.py --filter C3_best_ --out c3_best.csv"
