#!/usr/bin/env bash
# B9 + B8-MF IMVC eval — 4 protocols x 4 missing rates.
#
# Discovers ckpts under outputs/CRAFT_MVC/{B9_*,B8_mf_concat_mlp_*}/best.ckpt automatically.
# Dataset is inferred from exp_name (_hw_ -> handwritten, _cub_ -> cub, _mf_ -> multifashion).
# Idempotent: skips ckpts that already have four_protocol_eval_results.yaml.
#
# SetMLP / ConcatMLP do not have a mask_strategy parameter (only transformer does); eval uses the
# model's saved default config and does not pass --mask_strategy.
#
# Two-GPU fastest protocol (launch two terminals):
#   Terminal 1 (GPU 0): GPU=0 PARALLEL=5 PREFIXES="B9_hw_ B9_cub_"   bash scripts/ablations/run_eval_b9_setmlp.sh
#   Terminal 2 (GPU 1): GPU=1 PARALLEL=5 PREFIXES="B9_mf_ B8_mf_"    bash scripts/ablations/run_eval_b9_setmlp.sh
#
# When PREFIXES is unspecified, runs everything by default: B9_* + B8_mf_concat_mlp_*.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${PARALLEL:=5}"
: "${GPU:=0}"
: "${MISSING_RATES:=0.1 0.3 0.5 0.7}"
: "${NUM_TRIALS:=5}"
: "${EVAL_SEED:=42}"
: "${PROTOCOL:=all}"
: "${PREFIXES:=B9_ B8_mf_concat_mlp_}"

shopt -s nullglob
ckpts=()
for prefix in $PREFIXES; do
  ckpts+=( "$OUTPUT_ROOT"/${prefix}*/best.ckpt )
done
shopt -u nullglob

total=${#ckpts[@]}

if [[ $total -eq 0 ]]; then
  echo "No ckpts found under $OUTPUT_ROOT/{${PREFIXES// /,}}*/best.ckpt"
  echo "Training not finished? First run bash scripts/ablations/run_b9_setmlp.sh + run_b8_mf.sh"
  exit 1
fi

echo "B9 + B8-MF IMVC eval (4 protocols)"
echo "  prefixes     : $PREFIXES"
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
    *_mf_*)  local dataset=multifashion ;;
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
echo "  python aggregate_imvc_results.py --filter B9_,B8_mf_ --out imvc_b9_b8mf.csv"
echo "  python summarize_csv.py --csv imvc_b9_b8mf.csv --out b9_b8mf_imvc_summary.md"
