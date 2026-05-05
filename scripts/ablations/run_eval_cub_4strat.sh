#!/usr/bin/env bash
# C3 CUB 4-strategy IMVC eval — 5 ckpts x 4 mask_strategy.
#
# Same structure as the HW 4-strategy eval, but dataset=cub.
# Output layout:
#   outputs/CRAFT_MVC/C3_best_cub_s{42-46}/eval_{learnable,mean,attention_mask,zero}/...
#
# Idempotent: (seed, strategy) cells with existing yaml are skipped automatically -> resuming
# after interruption only fills missing ones.
#
# Total 20 tasks (5 seeds x 4 strategies), 4-protocol x 5 trials x 4 mr each.
# CUB V=2 + small dataset, single task is fast; 4 parallel takes ~30-60 min.
#
# Overrides:
#   PARALLEL=2 GPU=1   bash scripts/ablations/run_eval_cub_4strat.sh
#   STRATEGIES="learnable mean" bash ...                # only fill some strategies
#   DRY_RUN=1          bash ...

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${SEEDS:=42 43 44 45 46}"
: "${PARALLEL:=4}"
: "${GPU:=0}"
: "${MISSING_RATES:=0.1 0.3 0.5 0.7}"
: "${NUM_TRIALS:=5}"
: "${EVAL_SEED:=42}"
: "${PROTOCOL:=all}"
: "${STRATEGIES:=learnable mean attention_mask zero}"
: "${PREFIX:=C3_best_cub}"

n_seeds=$(echo "$SEEDS" | wc -w | tr -d ' ')
n_strats=$(echo "$STRATEGIES" | wc -w | tr -d ' ')
total=$(( n_seeds * n_strats ))

echo "CUB 4-strategy IMVC eval"
echo "  prefix    : $PREFIX"
echo "  seeds     : $SEEDS"
echo "  strategies: $STRATEGIES"
echo "  protocol  : $PROTOCOL"
echo "  mr        : $MISSING_RATES"
echo "  GPU=$GPU  parallel=$PARALLEL"
echo "  total tasks: $total (skipping already-done)"

eval_one() {
  local s="$1"; local strat="$2"
  local d="$OUTPUT_ROOT/${PREFIX}_s${s}"
  local ckpt="$d/best.ckpt"
  if [[ ! -f "$ckpt" ]]; then
    echo "    [SKIP] s$s: no ckpt at $ckpt"
    return 0
  fi
  local out_subdir="$d/eval_$strat"
  local out="$out_subdir/four_protocol_eval_results.yaml"
  local log="$LOG_DIR/eval_${PREFIX}_s${s}_${strat}.log"

  if [[ -f "$out" ]]; then
    echo "    [skip] s$s + $strat (yaml exists)"
    return 0
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "    [dry] s$s + $strat"
    return 0
  fi
  mkdir -p "$out_subdir"
  echo "    [eval gpu=$GPU] s$s + $strat -> $log"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$GPU" python incomplete_eval.py \
    --checkpoint "$ckpt" \
    --dataset cub \
    --data_dir "$DATA_DIR" \
    --protocol "$PROTOCOL" \
    --missing_rates $MISSING_RATES \
    --num_trials "$NUM_TRIALS" \
    --seed "$EVAL_SEED" \
    --mask_strategy "$strat" \
    --output_dir "$out_subdir" > "$log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "    [FAIL] s$s + $strat (exit $rc); see $log"
  fi
}

i=0
n_running=0
for s in $SEEDS; do
  for strat in $STRATEGIES; do
    i=$((i+1))
    while (( n_running >= PARALLEL )); do
      wait -n
      n_running=$((n_running-1))
    done
    eval_one "$s" "$strat" &
    n_running=$((n_running+1))
    echo "[$i/$total] launched s$s + $strat (in_flight=$n_running)"
  done
done

wait
echo "Eval done. Actual yaml count:"
echo "  $(find $OUTPUT_ROOT/${PREFIX}_s*/eval_*/four_protocol_eval_results.yaml 2>/dev/null | wc -l) / $total"
echo ""
echo "Compare 4 strategies:"
echo "  python scripts/compare_strategies.py --prefix C3_best cub"
