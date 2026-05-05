#!/usr/bin/env bash
# C3 IMVC eval — 4 protocols x 4 missing rates, run in parallel.
#
# Discovers ckpts under outputs/CRAFT_MVC/C3_best_*/best.ckpt automatically (10 ckpts).
# Dataset is inferred from exp_name (C3_best_hw_* / C3_best_cub_*).
# Idempotent: skips ckpts that already have four_protocol_eval_results.yaml.
#
# Default protocol=all (4 protocols), consistent with the b2b5b7 round; learning the lesson from
# the cub_imvc round of doing cell_level first and topping up later.
#
# Overrides:
#   PARALLEL=2     bash scripts/ablations/run_eval_best_5seeds.sh
#   GPU=1          bash scripts/ablations/run_eval_best_5seeds.sh
#   PROTOCOL=cell_level   bash scripts/ablations/run_eval_best_5seeds.sh
#   DRY_RUN=1      bash scripts/ablations/run_eval_best_5seeds.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${PARALLEL:=4}"
: "${GPU:=0}"
: "${MISSING_RATES:=0.1 0.3 0.5 0.7}"
: "${NUM_TRIALS:=5}"
: "${EVAL_SEED:=42}"
: "${PROTOCOL:=all}"

shopt -s nullglob
ckpts=( "$OUTPUT_ROOT"/C3_best_*/best.ckpt )
shopt -u nullglob
total=${#ckpts[@]}

if [[ $total -eq 0 ]]; then
  echo "No ckpts found under $OUTPUT_ROOT/C3_best_*/best.ckpt"
  echo "Training not finished? First run bash scripts/ablations/run_best_5seeds.sh"
  exit 1
fi

echo "C3 IMVC eval (4 protocols)"
echo "  ckpts found  : $total"
echo "  missing rates: $MISSING_RATES"
echo "  trials       : $NUM_TRIALS"
echo "  protocol     : $PROTOCOL"
echo "  GPU          : $GPU"
echo "  parallel     : $PARALLEL"

eval_one() {
  local ckpt="$1"
  local expdir="$(dirname "$ckpt")"
  local exp="$(basename "$expdir")"
  local out="$expdir/four_protocol_eval_results.yaml"
  local log="$LOG_DIR/eval_$exp.log"

  case "$exp" in
    *_hw_*)  local dataset=handwritten ;;
    *_cub_*) local dataset=cub ;;
    *)
      echo "    [SKIP] $exp - cannot infer dataset"
      return 0 ;;
  esac

  if [[ -f "$out" ]]; then
    echo "    [skip] $exp (results exists)"
    return 0
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "    [dry] $exp ($dataset)"
    return 0
  fi
  echo "    [eval gpu=$GPU] $exp ($dataset) -> $log"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$GPU" python incomplete_eval.py \
    --checkpoint "$ckpt" \
    --dataset "$dataset" \
    --data_dir "$DATA_DIR" \
    --protocol "$PROTOCOL" \
    --missing_rates $MISSING_RATES \
    --num_trials "$NUM_TRIALS" \
    --seed "$EVAL_SEED" > "$log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "    [FAIL] $exp (exit $rc); see $log"
  fi
}

i=0
n_running=0
for ckpt in "${ckpts[@]}"; do
  i=$((i+1))
  while (( n_running >= PARALLEL )); do
    wait -n
    n_running=$((n_running-1))
  done
  eval_one "$ckpt" &
  n_running=$((n_running+1))
  echo "[$i/$total] launched eval $(basename $(dirname $ckpt)) (in_flight=$n_running)"
done

echo "All $total launched. Waiting for last $n_running in flight..."
wait
echo "Eval done."
echo ""
echo "Aggregate:"
echo "  python aggregate_imvc_results.py --filter C3_best_ --out imvc_c3_best.csv"
echo "  python summarize_csv.py --csv imvc_c3_best.csv --out c3_best_imvc_summary.md"
