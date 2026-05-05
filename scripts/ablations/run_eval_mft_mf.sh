#!/usr/bin/env bash
# C5 IMVC eval — for each multifashion MFT ckpt, run a 4-protocol eval under each of the 4 mask_strategies.
#
# Same structure as run_eval_mft_hw.sh; only difference is dataset and PREFIX defaults.
#
# Output layout (per ckpt directory):
#   eval_learnable/four_protocol_eval_results.yaml
#   eval_mean/four_protocol_eval_results.yaml
#   eval_attention_mask/four_protocol_eval_results.yaml
#   eval_zero/four_protocol_eval_results.yaml
#
# After running:
#   python scripts/compare_strategies.py --prefix C5_mf_mft mf
#   python scripts/compare_strategies.py --prefix C2_mf_lr10_s_le05 mf   # baseline comparison
#
# Note: when compare_strategies.py uses dataset tag "mf", the glob would become C5_mf_mft_mf_*,
# but the actual exp name is C5_mf_mft_*. build_glob_pattern in the script detects that prefix
# already contains "_mf" and skips appending dataset, so this works correctly.

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
: "${PREFIX:=C5_mf_mft}"

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

echo "C5 eval: $PREFIX x ${STRATEGIES// /,}"
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
    --dataset multifashion \
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
echo "Compare 4 strategies on MFT ckpts:"
echo "  python scripts/compare_strategies.py --prefix C5_mf_mft mf"
echo ""
echo "Compare MFT vs C2 baseline (same strategy):"
echo "  python scripts/compare_strategies.py --prefix C2_mf_lr10_s_le05 mf"
