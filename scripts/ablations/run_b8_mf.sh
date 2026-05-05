#!/usr/bin/env bash
# B8-MF — ConcatMLPFusion x multifashion x 5 seeds.
#
# Fills the MF cell of the B8 spectrum (original run_b8.sh only covered hw + cub).
# Combined with B9 (set_mlp on MF) gives the middle column of the Section 6.3 spectrum:
# "C2 soft violation" vs "C2 strictly satisfied".
#
# Two-GPU split (recommended):
#   Terminal 1 (GPU 0): GPU=0 PARALLEL=5 bash scripts/ablations/run_b8_mf.sh
#   Pair with run_b9_setmlp.sh DATASETS="hw cub" running concurrently (GPU 0 runs two batches,
#   waves separated by a few GB of VRAM).
#
# Smoke:
#   SEEDS="42" PARALLEL=1 bash scripts/ablations/run_b8_mf.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${SEEDS:=42 43 44 45 46}"
: "${PARALLEL:=5}"
: "${GPU:=0}"

n_seeds=$(echo "$SEEDS" | wc -w | tr -d ' ')
total=$n_seeds

echo "B8-MF: ConcatMLPFusion on multifashion"
echo "  seeds    : $SEEDS"
echo "  GPU      : $GPU"
echo "  parallel : $PARALLEL"
echo "  total    : $total runs"

run_one() {
  local exp="$1"; shift
  local outdir="$OUTPUT_ROOT/$exp"
  local logfile="$LOG_DIR/$exp.log"
  if [[ -f "$outdir/results.yaml" ]]; then
    echo "    [skip] $exp (results.yaml exists)"
    return 0
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
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

base=$(mfashion_args)

i=0
n_running=0
for seed in $SEEDS; do
  i=$((i+1))
  exp="B8_mf_concat_mlp_s${seed}"

  while (( n_running >= PARALLEL )); do
    wait -n
    n_running=$((n_running-1))
  done

  # shellcheck disable=SC2086
  run_one "$exp" $base \
    --fusion_type concat_mlp \
    --seed "$seed" &
  n_running=$((n_running+1))
  echo "[$i/$total] launched $exp (in_flight=$n_running)"
done

echo "All $total launched. Waiting for last $n_running in flight..."
wait
echo "B8-MF training done."
echo ""
echo "Next: bash scripts/ablations/run_eval_b9_setmlp.sh   # eval handles B8/B9 both"
echo ""
echo "Aggregate training:"
echo "  python aggregate_results.py --filter 'B8_mf_' --out b8_mf.csv"
