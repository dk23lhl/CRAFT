#!/usr/bin/env bash
# Item #1: MF Transformer canonical re-eval (attention_mask).
#
# Run 4-protocol IMVC eval on C2_mf_lr10_s_le05_*/best.ckpt (multifashion complete-only training,
# 3 seeds 42/43/44) with --mask_strategy attention_mask, as the Transformer anchor for the
# Section 6.3 spectrum MF row. Complete-only training => mask_strategy does not enter forward =>
# eval-time override is clean (same setup as HW C3_best).
#
# Output: outputs/CRAFT_MVC/C2_mf_lr10_s_le05_s{seed}/eval_attention_mask/four_protocol_eval_results.yaml
#
# Run:
#   GPU=0 bash scripts/ablations/run_eval_mf_attn_canonical.sh
#
# Smoke:
#   SEEDS_FILTER="42" bash scripts/ablations/run_eval_mf_attn_canonical.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${PARALLEL:=3}"
: "${GPU:=0}"
: "${MISSING_RATES:=0.1 0.3 0.5 0.7}"
: "${NUM_TRIALS:=5}"
: "${EVAL_SEED:=42}"
: "${PROTOCOL:=all}"
: "${PREFIX:=C2_mf_lr10_s_le05}"
: "${SEEDS_FILTER:=}"  # empty = all; specify e.g. "42" to run a single seed

shopt -s nullglob
all_ckpts=( "$OUTPUT_ROOT"/${PREFIX}_*/best.ckpt )
shopt -u nullglob

ckpts=()
for c in "${all_ckpts[@]}"; do
  if [[ -z "$SEEDS_FILTER" ]]; then
    ckpts+=( "$c" )
  else
    for sd in $SEEDS_FILTER; do
      [[ "$c" == *_s${sd}/* ]] && ckpts+=( "$c" )
    done
  fi
done
total=${#ckpts[@]}

if [[ $total -eq 0 ]]; then
  echo "No ckpts under $OUTPUT_ROOT/${PREFIX}_*/best.ckpt"
  exit 1
fi

echo "Item #1: MF Transformer canonical re-eval (attention_mask)"
echo "  prefix       : $PREFIX"
echo "  ckpts        : $total"
echo "  missing rates: $MISSING_RATES"
echo "  trials       : $NUM_TRIALS"
echo "  protocol     : $PROTOCOL"
echo "  GPU          : $GPU"
echo "  parallel     : $PARALLEL"

eval_one() {
  local ckpt="$1"
  local expdir="$(dirname "$ckpt")"
  local exp="$(basename "$expdir")"
  local out_subdir="$expdir/eval_attention_mask"
  local out="$out_subdir/four_protocol_eval_results.yaml"
  local log="$LOG_DIR/eval_${exp}_attention_mask.log"

  if [[ -f "$out" ]]; then
    echo "    [skip] $exp (eval_attention_mask exists)"
    return 0
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "    [dry] $exp"
    return 0
  fi
  mkdir -p "$out_subdir"
  echo "    [eval gpu=$GPU] $exp -> $log"
  CUDA_VISIBLE_DEVICES="$GPU" python incomplete_eval.py \
    --checkpoint "$ckpt" \
    --dataset multifashion \
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
  echo "[$i/$total] launched eval $(basename $(dirname $ckpt))"
done

echo "Waiting for $n_running in flight..."
wait
echo "MF canonical re-eval done."
echo ""
echo "Aggregate:"
echo "  python aggregate_imvc_results.py --filter ${PREFIX}_ --eval_subdir eval_attention_mask --out imvc_mf_attn.csv"
echo "  python summarize_csv.py --csv imvc_mf_attn.csv --out mf_attn_summary.md"
