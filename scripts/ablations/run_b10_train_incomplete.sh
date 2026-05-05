#!/usr/bin/env bash
# B10 — Rate-matched control under Protocol 4 (used by CUB Appx D.6).
#
# Default 20 runs: 1 dataset (cub) x 4 missing rates x 5 seeds.
#   datasets : cub
#   rates    : 0.1, 0.3, 0.5, 0.7
#   seeds    : 42, 43, 44, 45, 46
#
# Key: training mask is Protocol 4 (cell_level Bernoulli), same distribution as
# incomplete_eval.py's `cell_level` protocol; eval also only runs the matched (P4, r) cell
# (see run_eval_b10.sh).
#
# Training parameters:
#   --missing_rate r                : cell-level missing rate r
#   --train_mask_protocol cell_level: Protocol 4 (cell-level Bernoulli)
#   --mask_strategy attention_mask  : at forward time, mask missing views via key_padding_mask
#   --skip_diag                     : skip random encoder diagnostic
#
# Two-GPU protocol (CUB is small, single-GPU PARALLEL=4 usually suffices; to split GPUs, partition by RATES):
#   Terminal 1 (GPU 0): GPU=0 PARALLEL=4 RATES="0.1 0.3" bash scripts/ablations/run_b10_train_incomplete.sh
#   Terminal 2 (GPU 1): GPU=1 PARALLEL=4 RATES="0.5 0.7" bash scripts/ablations/run_b10_train_incomplete.sh
#
# Single GPU (recommended, CUB is light, all 20 runs ~30 min w/ PARALLEL=4):
#   GPU=0 PARALLEL=4 bash scripts/ablations/run_b10_train_incomplete.sh
#
# Smoke:
#   SEEDS="42" RATES="0.5" PARALLEL=1 bash scripts/ablations/run_b10_train_incomplete.sh
#
# Overrides:
#   DATASETS="hw cub"      bash ...        # also run HW (e.g. if D.6 is later extended to HW)
#   SEEDS="42 43 44"       bash ...        # 3 seeds (validate first)
#   DRY_RUN=1              bash ...        # print commands only

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${SEEDS:=42 43 44 45 46}"
: "${RATES:=0.1 0.3 0.5 0.7}"
: "${DATASETS:=cub}"
: "${MASK_STRATEGY:=attention_mask}"
: "${TRAIN_MASK_PROTOCOL:=cell_level}"
: "${PARALLEL:=4}"
: "${GPU:=0}"

# Protocol tag derived automatically (for exp naming, avoids P1/P4 overwriting each other).
case "$TRAIN_MASK_PROTOCOL" in
  cell_level) PROTOCOL_TAG="p4" ;;
  completer)  PROTOCOL_TAG="p1" ;;
  *) echo "Unknown TRAIN_MASK_PROTOCOL=$TRAIN_MASK_PROTOCOL"; exit 1 ;;
esac

n_seeds=$(echo "$SEEDS" | wc -w | tr -d ' ')
n_rates=$(echo "$RATES" | wc -w | tr -d ' ')
n_ds=$(echo "$DATASETS" | wc -w | tr -d ' ')
total=$(( n_ds * n_rates * n_seeds ))

echo "B10: rate-matched control"
echo "  datasets       : $DATASETS"
echo "  missing rates  : $RATES"
echo "  seeds          : $SEEDS"
echo "  train protocol : $TRAIN_MASK_PROTOCOL  (exp tag = ${PROTOCOL_TAG})"
echo "  mask_strategy  : $MASK_STRATEGY"
echo "  GPU            : $GPU"
echo "  parallel       : $PARALLEL"
echo "  total runs     : $total"

# rate=0.1 -> mr_tag=010; rate=0.5 -> mr_tag=050
fmt_rate_tag() {
  local r="$1"
  printf "%03d" "$(echo "$r * 100" | bc | awk '{printf "%d", $1+0.5}')"
}

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

i=0
n_running=0
for dataset in $DATASETS; do
  base=$(${dataset}_args)
  for rate in $RATES; do
    mr_tag=$(fmt_rate_tag "$rate")
    for seed in $SEEDS; do
      i=$((i+1))
      exp="B10${PROTOCOL_TAG}_${dataset}_mr${mr_tag}_s${seed}"

      while (( n_running >= PARALLEL )); do
        wait -n
        n_running=$((n_running-1))
      done

      # shellcheck disable=SC2086
      run_one "$exp" $base \
        --missing_rate "$rate" \
        --train_mask_protocol "$TRAIN_MASK_PROTOCOL" \
        --mask_strategy "$MASK_STRATEGY" \
        --skip_diag \
        --seed "$seed" &
      n_running=$((n_running+1))
      echo "[$i/$total] launched $exp (in_flight=$n_running)"
    done
  done
done

echo "All $total launched. Waiting for last $n_running in flight..."
wait
echo "B10 training done."
echo ""
echo "Next: bash scripts/ablations/run_eval_b10.sh"
echo ""
echo "Aggregate (training):"
echo "  python aggregate_results.py --filter B10_ --out b10.csv"
echo "  python summarize_csv.py --csv b10.csv --out b10_summary.md"
