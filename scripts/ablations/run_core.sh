#!/usr/bin/env bash
# Core — CRAFT-Core ablation: Full minus MFT minus SimSiam minus entropy minus KL.
#
# CRAFT-Core only keeps: encoder + fusion transformer + reconstruction loss + Stage 2 K-Means init.
# All clustering-side losses and MFT are off:
#   --lambda_repr 0       disables SimSiam (representation alignment)
#   --lambda_entropy 0    disables entropy regularization
#   --lambda_cluster 0    disables KL (cluster consistency)
#   --train_mask_prob not passed; default 0, disables MFT
#   --lambda_recon 1.0    default, reconstruction kept
#
# Therefore Stage 1 = recon only, Stage 2 transition = K-Means init, Stage 2 = recon only
# (cluster_head receives no gradient, final head weights = K-Means centroids).
#
# Total 6 runs: 2 datasets (hw, cub) x 3 seeds.
#
# Overrides:
#   SEEDS="42" PARALLEL=1 bash scripts/ablations/run_core.sh    # 1-seed smoke
#   GPU=1 PARALLEL=2 bash scripts/ablations/run_core.sh
#   DRY_RUN=1 bash scripts/ablations/run_core.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${SEEDS:=42 43 44}"
: "${PARALLEL:=2}"
: "${GPU:=0}"

datasets=(hw cub)
n_seeds=$(echo "$SEEDS" | wc -w | tr -d ' ')
total=$(( ${#datasets[@]} * n_seeds ))

echo "Core: CRAFT-Core ablation (recon-only)"
echo "  datasets : ${datasets[*]}"
echo "  seeds    : $SEEDS"
echo "  GPU      : $GPU"
echo "  parallel : $PARALLEL"
echo "  total    : $total runs"

run_one() {
  local exp="$1"; shift
  local outdir="$OUTPUT_ROOT/$exp"
  local logfile="$LOG_DIR/$exp.log"

  if [[ -f "$outdir/results.yaml" ]]; then
    echo "    [skip] $exp (results exists)"
    return 0
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "    [dry gpu=$GPU] $exp $*"
    return 0
  fi
  echo "    [run gpu=$GPU] $exp"
  CUDA_VISIBLE_DEVICES="$GPU" python main_mvc.py "$@" \
    --exp_name "$exp" \
    --output_dir "$(dirname "$OUTPUT_ROOT")" \
    > "$logfile" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "    [FAIL] $exp (exit $rc); see $logfile"
  fi
}

i=0
n_running=0

for dataset in "${datasets[@]}"; do
  base=$(${dataset}_args)
  for seed in $SEEDS; do
    i=$((i+1))
    exp="Core_${dataset}_s${seed}"

    while (( n_running >= PARALLEL )); do
      wait -n
      n_running=$((n_running-1))
    done

    # shellcheck disable=SC2086
    run_one "$exp" \
      $base \
      --lambda_repr 0 \
      --lambda_entropy 0 \
      --lambda_cluster 0 \
      --seed "$seed" &
    n_running=$((n_running+1))
    echo "[$i/$total] launched $exp (in_flight=$n_running)"
  done
done

echo "All $total launched. Waiting for $n_running in flight..."
wait

echo "Core training done."
echo ""
echo "Aggregate:"
echo "  python aggregate_results.py --filter Core_ --out core.csv"
