#!/usr/bin/env bash
# IMVC eval for B2 + B5 + B7 ckpts.
#
# Discovers ckpts under outputs/CRAFT_MVC/{B2_,B5_,B7_}*/best.ckpt automatically;
# dataset is inferred from exp_name (*_hw_* / *_cub_*).
# Idempotent: skips ckpts that already have four_protocol_eval_results.yaml.
#
# Default protocol=all (run all 4 protocols) — fill the reviewer table in one shot;
# don't repeat the CUB pattern of doing cell_level first and topping up later.
#
# Overrides:
#   PROTOCOL=cell_level     bash scripts/ablations/run_eval_b2b5b7.sh   # only run cell_level
#   MISSING_RATES="0.3 0.5" bash scripts/ablations/run_eval_b2b5b7.sh
#   NUM_TRIALS=3            bash scripts/ablations/run_eval_b2b5b7.sh
#   DRY_RUN=1               bash scripts/ablations/run_eval_b2b5b7.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${MISSING_RATES:=0.1 0.3 0.5 0.7}"
: "${NUM_TRIALS:=5}"
: "${EVAL_SEED:=42}"
: "${PROTOCOL:=all}"

shopt -s nullglob
ckpts=(
  "$OUTPUT_ROOT"/B2_*/best.ckpt
  "$OUTPUT_ROOT"/B5_*/best.ckpt
  "$OUTPUT_ROOT"/B7_*/best.ckpt
)
shopt -u nullglob

total=${#ckpts[@]}
if [[ $total -eq 0 ]]; then
  echo "No ckpts found under $OUTPUT_ROOT/{B2_,B5_,B7_}*/best.ckpt"
  exit 1
fi

echo "IMVC eval for B2 + B5 + B7"
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

  if [[ "$DRY_RUN" == "1" ]]; then
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
echo "Aggregate:"
echo "  python aggregate_imvc_results.py --filter B2_,B5_,B7_ --out imvc_b2b5b7.csv"
echo "  python summarize_csv.py --csv imvc_b2b5b7.csv --out b2b5b7_imvc_summary.md"
