#!/usr/bin/env bash
# Cell-Level + 4-protocol IMVC eval for Core ckpts.
#
# Discovers ckpts under outputs/CRAFT_MVC/Core_*/best.ckpt automatically; dataset inferred from exp_name.
# Idempotent: skips ckpts that already have four_protocol_eval_results.yaml.
#
# Overrides:
#   MISSING_RATES="0.3 0.5"  bash scripts/ablations/run_eval_core.sh
#   NUM_TRIALS=3             bash scripts/ablations/run_eval_core.sh
#   PROTOCOL=cell_level      bash scripts/ablations/run_eval_core.sh   # only run cell_level
#   DRY_RUN=1                bash scripts/ablations/run_eval_core.sh

set -uo pipefail   # don't use -e: a single ckpt failing should not interrupt the whole batch

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${MISSING_RATES:=0.1 0.3 0.5 0.7}"
: "${NUM_TRIALS:=5}"
: "${EVAL_SEED:=42}"
: "${PROTOCOL:=all}"

shopt -s nullglob
ckpts=( "$OUTPUT_ROOT"/Core_*/best.ckpt )
shopt -u nullglob

total=${#ckpts[@]}
if [[ $total -eq 0 ]]; then
  echo "No ckpts found under $OUTPUT_ROOT/Core_*/best.ckpt"
  exit 1
fi

echo "IMVC eval for Core"
echo "  ckpts found   : $total"
echo "  missing rates : $MISSING_RATES"
echo "  trials        : $NUM_TRIALS"
echo "  eval seed     : $EVAL_SEED"
echo "  protocol      : $PROTOCOL"

i=0
n_done=0
n_skip=0
n_fail=0

for ckpt in "${ckpts[@]}"; do
  i=$((i+1))
  expdir="$(dirname "$ckpt")"
  exp="$(basename "$expdir")"

  case "$exp" in
    *_hw_*)  dataset=handwritten ;;
    *_cub_*) dataset=cub ;;
    *)
      echo "[$i/$total] [SKIP] $exp - cannot infer dataset"
      n_skip=$((n_skip+1))
      continue
      ;;
  esac

  results_file="$expdir/four_protocol_eval_results.yaml"
  logfile="$LOG_DIR/eval_$exp.log"

  if [[ -f "$results_file" ]]; then
    echo "[$i/$total] [skip] $exp (results exists)"
    n_skip=$((n_skip+1))
    continue
  fi

  echo "[$i/$total] [eval] $exp ($dataset) → $logfile"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "    [dry] python incomplete_eval.py --checkpoint $ckpt --dataset $dataset --protocol $PROTOCOL --missing_rates $MISSING_RATES --num_trials $NUM_TRIALS --seed $EVAL_SEED"
    continue
  fi

  # shellcheck disable=SC2086
  python incomplete_eval.py \
    --checkpoint "$ckpt" \
    --dataset "$dataset" \
    --data_dir "$DATA_DIR" \
    --protocol "$PROTOCOL" \
    --missing_rates $MISSING_RATES \
    --num_trials "$NUM_TRIALS" \
    --seed "$EVAL_SEED" > "$logfile" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "    [FAIL] $exp (exit $rc); see $logfile"
    n_fail=$((n_fail+1))
  else
    n_done=$((n_done+1))
  fi
done

echo "Eval done. ok=$n_done  skip=$n_skip  fail=$n_fail  total=$total"
echo ""
echo "Aggregate + summary:"
echo "  python aggregate_results.py      --filter Core_ --out core_train.csv"
echo "  python aggregate_imvc_results.py --filter Core_ --out core_imvc.csv"
echo "  python summarize_csv.py --csv core_train.csv core_imvc.csv --out core_summary.md"
