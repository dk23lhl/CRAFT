#!/usr/bin/env bash
# Item #3: HW + CUB C3 canonical re-eval (attention_mask).
#
# Re-run 4-protocol eval on C3_best_hw_*/best.ckpt and C3_best_cub_*/best.ckpt with
# --mask_strategy attention_mask. These two batches were trained complete-only (mask_strategy
# does not enter forward), so eval-time override is a clean switch.
#
# Idempotent: skips ckpts that already have eval_attention_mask/four_protocol_eval_results.yaml.
#
# Two-GPU fastest protocol:
#   Terminal 1: GPU=0 PARALLEL=5 PREFIX=C3_best_hw_  bash scripts/ablations/run_eval_c3_attn.sh
#   Terminal 2: GPU=1 PARALLEL=5 PREFIX=C3_best_cub_ bash scripts/ablations/run_eval_c3_attn.sh
#
# Single GPU sequential:
#   GPU=0 PARALLEL=5 bash scripts/ablations/run_eval_c3_attn.sh   # runs both HW + CUB

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
: "${PREFIX:=}"  # empty = run both HW + CUB; specify e.g. C3_best_hw_ to run HW only

shopt -s nullglob
ckpts=()
if [[ -z "$PREFIX" ]]; then
  ckpts+=( "$OUTPUT_ROOT"/C3_best_hw_*/best.ckpt )
  ckpts+=( "$OUTPUT_ROOT"/C3_best_cub_*/best.ckpt )
else
  ckpts+=( "$OUTPUT_ROOT"/${PREFIX}*/best.ckpt )
fi
shopt -u nullglob
total=${#ckpts[@]}

if [[ $total -eq 0 ]]; then
  echo "No ckpts found"
  exit 1
fi

echo "Item #3: C3 canonical re-eval (attention_mask)"
echo "  ckpts        : $total"
echo "  prefix       : ${PREFIX:-C3_best_hw_ + C3_best_cub_}"
echo "  missing rates: $MISSING_RATES"
echo "  GPU          : $GPU  parallel: $PARALLEL"

eval_one() {
  local ckpt="$1"
  local expdir="$(dirname "$ckpt")"
  local exp="$(basename "$expdir")"
  local out_subdir="$expdir/eval_attention_mask"
  local out="$out_subdir/four_protocol_eval_results.yaml"
  local log="$LOG_DIR/eval_${exp}_attention_mask.log"

  case "$exp" in
    *_hw_*)  local dataset=handwritten ;;
    *_cub_*) local dataset=cub ;;
    *)
      echo "    [SKIP] $exp - cannot infer dataset"
      return 0 ;;
  esac

  if [[ -f "$out" ]]; then
    echo "    [skip] $exp (already done)"
    return 0
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "    [dry] $exp ($dataset)"
    return 0
  fi
  mkdir -p "$out_subdir"
  echo "    [eval gpu=$GPU] $exp ($dataset) -> $log"
  CUDA_VISIBLE_DEVICES="$GPU" python incomplete_eval.py \
    --checkpoint "$ckpt" \
    --dataset "$dataset" \
    --data_dir "$DATA_DIR" \
    --protocol "$PROTOCOL" \
    --missing_rates $MISSING_RATES \
    --num_trials "$NUM_TRIALS" \
    --seed "$EVAL_SEED" \
    --mask_strategy attention_mask \
    --output_dir "$out_subdir" > "$log" 2>&1
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
  echo "[$i/$total] launched"
done

echo "Waiting for $n_running in flight..."
wait
echo "C3 attention_mask re-eval done."
echo ""
echo "Aggregate:"
echo "  python aggregate_imvc_results.py --filter C3_best_ --eval_subdir eval_attention_mask --out imvc_c3_attn.csv"
echo "  python summarize_csv.py --csv imvc_c3_attn.csv --out c3_attn_summary.md"
