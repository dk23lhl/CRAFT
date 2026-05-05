#!/usr/bin/env bash
# Cell-Level IMVC eval for B8 + A2 ckpts.
#
# B8 (20 ckpts): transformer vs concat_mlp x {hw, cub} x seeds 42-46
#   -> verify C1/C2: ConcatMLPFusion should degrade noticeably under missing views.
# A2 (12 ckpts): {learnable, zero, mean, attention_mask} x cub x seeds 42-44
#   -> mask_strategy difference under complete training + missing inference.
#
# Discovers ckpts under outputs/CRAFT_MVC/{B8_,A2_}*/best.ckpt automatically; dataset inferred from exp_name.
# Idempotent: skips ckpts that already have four_protocol_eval_results.yaml.
#
# Overrides:
#   MISSING_RATES="0.3 0.5"  bash scripts/ablations/run_eval_b8_a2.sh
#   NUM_TRIALS=3             bash scripts/ablations/run_eval_b8_a2.sh
#   PROTOCOL=all             bash scripts/ablations/run_eval_b8_a2.sh   # all 4 protocols
#   DRY_RUN=1                bash scripts/ablations/run_eval_b8_a2.sh

set -uo pipefail   # don't use -e: a single ckpt failing should not interrupt the whole batch

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${MISSING_RATES:=0.1 0.3 0.5 0.7}"
: "${NUM_TRIALS:=5}"
: "${EVAL_SEED:=42}"
: "${PROTOCOL:=cell_level}"

shopt -s nullglob
ckpts=(
  "$OUTPUT_ROOT"/B8_*/best.ckpt
  "$OUTPUT_ROOT"/A2_*/best.ckpt
  "$OUTPUT_ROOT"/B8v2_*/best.ckpt
  "$OUTPUT_ROOT"/A2v2_*/best.ckpt
)
shopt -u nullglob

total=${#ckpts[@]}
if [[ $total -eq 0 ]]; then
  echo "No ckpts found under $OUTPUT_ROOT/{B8_,A2_,B8v2_,A2v2_}*/best.ckpt"
  exit 1
fi

echo "Cell-Level IMVC eval for B8 + A2"
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
echo "Aggregate with: python aggregate_imvc_results.py --filter B8_,A2_ --out imvc.csv"
