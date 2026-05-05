#!/usr/bin/env bash
# C4 IMVC eval — for each MFT-trained ckpt, run a 4-protocol eval under each of the 4 mask_strategies.
#
# Key: during MFT training, mask_token finally gets triggered in forward -> learnable should
# now work. Evaluate under all 4 strategies to see whether MFT really lets learnable catch up with
# or surpass mean/attention_mask.
#
# Output layout (per ckpt directory):
#   eval_learnable/four_protocol_eval_results.yaml
#   eval_mean/four_protocol_eval_results.yaml
#   eval_attention_mask/four_protocol_eval_results.yaml
#   eval_zero/four_protocol_eval_results.yaml
#
# After running, use scripts/compare_strategies.py to compare the 4 strategies; cross-prefix C3
# vs C4 comparison is also possible (C3 = complete-only training, C4 = MFT).
#
# Overrides:
#   PARALLEL=2 GPU=1 bash scripts/ablations/run_eval_mft_hw.sh
#   STRATEGIES="learnable mean" bash ...        # only two strategies
#   PROTOCOL=cell_level bash ...                # only one protocol (faster)

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
: "${STRATEGIES:=learnable mean attention_mask zero}"
: "${PREFIX:=C4_hw_mft}"

shopt -s nullglob
ckpts=( "$OUTPUT_ROOT"/${PREFIX}_*/best.ckpt )
shopt -u nullglob
n_ckpts=${#ckpts[@]}

if [[ $n_ckpts -eq 0 ]]; then
  echo "No ckpts under $OUTPUT_ROOT/${PREFIX}_*/best.ckpt"
  exit 1
fi

n_strats=$(echo "$STRATEGIES" | wc -w | tr -d ' ')
total=$(( n_ckpts * n_strats ))

echo "C4 eval: $PREFIX x ${STRATEGIES// /,}"
echo "  ckpts        : $n_ckpts"
echo "  strategies   : $STRATEGIES"
echo "  missing_rates: $MISSING_RATES"
echo "  protocol     : $PROTOCOL"
echo "  total tasks  : $total"
echo "  GPU=$GPU  parallel=$PARALLEL"

eval_one() {
  local ckpt="$1"; local strat="$2"
  local expdir="$(dirname "$ckpt")"
  local exp="$(basename "$expdir")"
  local out_subdir="$expdir/eval_$strat"
  local out="$out_subdir/four_protocol_eval_results.yaml"
  local log="$LOG_DIR/eval_${exp}_${strat}.log"

  if [[ -f "$out" ]]; then
    echo "    [skip] $exp + $strat"
    return 0
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "    [dry] $exp + $strat"
    return 0
  fi
  mkdir -p "$out_subdir"
  echo "    [eval gpu=$GPU] $exp + $strat -> $log"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$GPU" python incomplete_eval.py \
    --checkpoint "$ckpt" \
    --dataset handwritten \
    --data_dir "$DATA_DIR" \
    --protocol "$PROTOCOL" \
    --missing_rates $MISSING_RATES \
    --num_trials "$NUM_TRIALS" \
    --seed "$EVAL_SEED" \
    --mask_strategy "$strat" \
    --output_dir "$out_subdir" > "$log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "    [FAIL] $exp + $strat (exit $rc); see $log"
  fi
}

i=0
n_running=0
for ckpt in "${ckpts[@]}"; do
  for strat in $STRATEGIES; do
    i=$((i+1))
    while (( n_running >= PARALLEL )); do
      wait -n
      n_running=$((n_running-1))
    done
    eval_one "$ckpt" "$strat" &
    n_running=$((n_running+1))
    echo "[$i/$total] launched $(basename $(dirname $ckpt)) + $strat (in_flight=$n_running)"
  done
done

wait
echo "Eval done."
echo ""
echo "Compare 4 strategies (single prefix):"
echo "  python scripts/compare_strategies.py hw       # default reads C3_best_hw_*"
echo ""
echo "Compare C3 vs C4 (single strategy): adjust the prefix in compare_strategies.py temporarily,"
echo "or read yaml directly:"
echo "  for s in 42 43 44 45 46; do"
echo "    f1=outputs/CRAFT_MVC/C3_best_hw_s\$s/eval_learnable/four_protocol_eval_results.yaml"
echo "    f2=outputs/CRAFT_MVC/C4_hw_mft_p05_s\$s/eval_learnable/four_protocol_eval_results.yaml"
echo "    echo \"seed=\$s C3=\$(...) C4=\$(...)\""
echo "  done"
