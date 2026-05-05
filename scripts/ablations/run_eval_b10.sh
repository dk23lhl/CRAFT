#!/usr/bin/env bash
# B10 IMVC eval — matched (P4, r) only.
#
# Key constraint: each ckpt is only evaluated on the (Protocol 4, rate r) cell it was trained on.
# Unlike old scripts, this does not run the full 4 protocol x 4 rate grid.
# ckpt name B10_<ds>_mr<XYZ>_s<seed>; rate is parsed from mr<XYZ>.
#
# Eval parameters:
#   --protocol cell_level     : Protocol 4, same distribution as training mask
#   --missing_rates <matched> : single rate (parsed from ckpt name)
#   --num_trials 5 --seed 42
#
# Eval seed defaults to 42 (same value as training missing_seed, but they use independent RNG streams,
# masks are uncorrelated). If concerned, set EVAL_SEED=100.
#
# Two-GPU protocol (split by RATES):
#   Terminal 1 (GPU 0): GPU=0 PARALLEL=4 RATES="0.1 0.3" bash scripts/ablations/run_eval_b10.sh
#   Terminal 2 (GPU 1): GPU=1 PARALLEL=4 RATES="0.5 0.7" bash scripts/ablations/run_eval_b10.sh
#
# Single GPU:
#   GPU=0 PARALLEL=4 bash scripts/ablations/run_eval_b10.sh
#
# Overrides:
#   NUM_TRIALS=3   bash ...
#   EVAL_SEED=100  bash ...
#   PREFIXES="B10_" bash ...   # default is B10_
#   DRY_RUN=1      bash ...

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${PARALLEL:=4}"
: "${GPU:=0}"
: "${NUM_TRIALS:=5}"
: "${EVAL_SEED:=42}"
: "${PREFIXES:=B10p1_ B10p4_}"
: "${RATES:=0.1 0.3 0.5 0.7}"   # filter which ckpts go into eval (matched by mr_tag)

# rate=0.1 -> mr_tag=010; rate=0.5 -> mr_tag=050
fmt_rate_tag() {
  local r="$1"
  printf "%03d" "$(echo "$r * 100" | bc | awk '{printf "%d", $1+0.5}')"
}

# Filter by RATES: only ckpts whose mr_tag belongs to the RATES set.
ALLOWED_TAGS=()
for r in $RATES; do
  ALLOWED_TAGS+=("$(fmt_rate_tag "$r")")
done

shopt -s nullglob
ckpts=()
for prefix in $PREFIXES; do
  for ckpt in "$OUTPUT_ROOT"/${prefix}*/best.ckpt; do
    exp="$(basename "$(dirname "$ckpt")")"
    # Parse mr_tag from exp name (B10_<ds>_mr<XYZ>_s<seed>).
    if [[ "$exp" =~ _mr([0-9]+)_s ]]; then
      tag="${BASH_REMATCH[1]}"
      for allowed in "${ALLOWED_TAGS[@]}"; do
        if [[ "$tag" == "$allowed" ]]; then
          ckpts+=("$ckpt")
          break
        fi
      done
    fi
  done
done
shopt -u nullglob

total=${#ckpts[@]}

if [[ $total -eq 0 ]]; then
  echo "No B10 ckpts found matching prefixes=$PREFIXES rates=$RATES"
  echo "Run training first: bash scripts/ablations/run_b10_train_incomplete.sh"
  exit 1
fi

echo "B10 IMVC eval (matched cell only, protocol auto-routed)"
echo "  prefixes      : $PREFIXES"
echo "  rates         : $RATES"
echo "  ckpts found   : $total"
echo "  trials        : $NUM_TRIALS"
echo "  eval seed     : $EVAL_SEED"
echo "  protocol routing:"
echo "                  B10p1_* -> completer (P1)"
echo "                  B10p4_* -> cell_level (P4)"
echo "                  B10_*   -> cell_level (legacy, treated as P4)"
echo "  GPU           : $GPU"
echo "  parallel      : $PARALLEL"

# Recover rate from mr_tag (010 -> 0.1, 050 -> 0.5).
tag_to_rate() {
  local tag="$1"
  awk -v t="$tag" 'BEGIN { printf "%g", t/100 }'
}

eval_one() {
  local ckpt="$1"
  local expdir="$(dirname "$ckpt")"
  local exp="$(basename "$expdir")"
  local out="$expdir/four_protocol_eval_results.yaml"
  local log="$LOG_DIR/eval_$exp.log"

  case "$exp" in
    *_hw_*)  local dataset=handwritten ;;
    *_cub_*) local dataset=cub ;;
    *_mf_*)  local dataset=multifashion ;;
    *)
      echo "    [SKIP] $exp - cannot infer dataset"
      return 0 ;;
  esac

  # Infer training protocol from exp prefix -> matched eval protocol.
  local eval_protocol
  case "$exp" in
    B10p1_*) eval_protocol=completer ;;
    B10p4_*) eval_protocol=cell_level ;;
    B10_*)   eval_protocol=cell_level ;;   # legacy naming (no protocol tag); treat as P4
    *)
      echo "    [SKIP] $exp - cannot infer protocol from prefix"
      return 0 ;;
  esac

  # Parse matched rate.
  if [[ ! "$exp" =~ _mr([0-9]+)_s ]]; then
    echo "    [SKIP] $exp - cannot parse mr_tag from name"
    return 0
  fi
  local matched_rate
  matched_rate=$(tag_to_rate "${BASH_REMATCH[1]}")

  if [[ -f "$out" ]]; then
    echo "    [skip] $exp (results exists)"
    return 0
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "    [dry gpu=$GPU] $exp ($dataset, $eval_protocol, matched_rate=$matched_rate)"
    return 0
  fi
  echo "    [eval gpu=$GPU] $exp ($dataset, $eval_protocol, matched_rate=$matched_rate) -> $log"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$GPU" python incomplete_eval.py \
    --checkpoint "$ckpt" \
    --dataset "$dataset" \
    --data_dir "$DATA_DIR" \
    --protocol "$eval_protocol" \
    --missing_rates "$matched_rate" \
    --num_trials "$NUM_TRIALS" \
    --seed "$EVAL_SEED" > "$log" 2>&1
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
  echo "[$i/$total] launched eval $(basename $(dirname $ckpt)) (in_flight=$n_running)"
done

echo "All $total launched. Waiting for last $n_running in flight..."
wait
echo "Eval done."
echo ""
echo "Aggregate:"
echo "  python aggregate_imvc_results.py --filter B10_ --out imvc_b10.csv"
echo "  python summarize_csv.py --csv imvc_b10.csv --out b10_imvc_summary.md"
