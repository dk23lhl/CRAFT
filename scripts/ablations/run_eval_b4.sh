#!/usr/bin/env bash
# B4 IMVC eval — Stage / Loss ablation under missing views.
#
# Motivation: B4 training (complete views) shows stage2 + entropy contribution ~ 0 (diff < 0.001).
# But the paper's selling point is the two-stage design, so we need to retest under IMVC —
# see whether stage2 only matters under missing views ("two-stage serves robustness"). If there
# is also no difference under IMVC, stage2 should be downgraded to an optional module in the paper.
#
# All 36 ckpts evaluated: 6 variants x 2 datasets x 3 seeds.
#   variants: full, no_entropy_s2, no_pretrain, no_stage2, recon_only_s1, repr_only_s1
#
# Key comparisons (IMVC mr=0.5 acc):
#   full vs no_stage2     -> is stage2 necessary under missing views?
#   full vs no_entropy_s2 -> is entropy necessary under missing views?
#   recon_only_s1 (CUB)   -> already +18pp at training time, does the lead hold under IMVC?
#   repr_only_s1 (HW)     -> already collapsed to 0.19 at training, used as control.
#
# Discovers ckpts under outputs/CRAFT_MVC/B4_*/best.ckpt automatically; dataset inferred from exp_name.
# Idempotent: skips ckpts that already have four_protocol_eval_results.yaml.
#
# Overrides:
#   MISSING_RATES="0.3 0.5"  bash scripts/ablations/run_eval_b4.sh
#   NUM_TRIALS=3             bash scripts/ablations/run_eval_b4.sh
#   DRY_RUN=1                bash scripts/ablations/run_eval_b4.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${MISSING_RATES:=0.1 0.3 0.5 0.7}"
: "${NUM_TRIALS:=5}"
: "${EVAL_SEED:=42}"
: "${PROTOCOL:=cell_level}"

shopt -s nullglob
ckpts=("$OUTPUT_ROOT"/B4_*/best.ckpt)
shopt -u nullglob

total=${#ckpts[@]}
if [[ $total -eq 0 ]]; then
  echo "No ckpts found under $OUTPUT_ROOT/B4_*/best.ckpt"
  exit 1
fi

echo "Cell-Level IMVC eval for B4 (stage/loss ablation)"
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

echo "B4 eval done. ok=$n_done  skip=$n_skip  fail=$n_fail  total=$total"
echo ""
echo "Aggregate:"
echo "  python aggregate_imvc_results.py --filter B4_ --out imvc_b4.csv"
echo "  python summarize_csv.py --csv imvc_b4.csv --out b4_imvc_summary.md"
