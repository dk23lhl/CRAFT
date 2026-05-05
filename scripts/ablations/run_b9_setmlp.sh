#!/usr/bin/env bash
# B9 — SetMLPFusion (DeepSets-style, strictly satisfies C2) x {HW, CUB, MF} x 5 seeds.
#
# Relation to B8 (transformer vs concat_mlp): B8 validates "C2 soft violation (concat) << C1+C2
# (transformer)"; B9 adds "C1+C2 strictly satisfied but simpler architecture (set_mlp)" as the
# second witness of the Section 6.3 spectrum.
#
# Capacity-matched hidden (solved from the ConcatMLP parameter budget 2dh + 2h^2 at d=128 baseline):
#   HW : 192  (~122k params)
#   CUB: 128  (~66k params)
#   MF : 144  (~82k params)
# If the main-table baseline actually has embed_dim != 128 (hw_args/mfashion_args are actually 256),
# switch to "based on actual baseline ConcatMLP capacity" via:
#   SETMLP_HIDDEN_HW=560 SETMLP_HIDDEN_MF=400 SETMLP_HIDDEN_CUB=168
#
# Parallelism: default PARALLEL=5 (matches the fastest path for 2-GPU x 5-slot).
# Use GPU + PARALLEL + DATASETS env vars to shard the workload.
#
# Two-GPU fastest protocol:
#   Terminal 1 (GPU 0): GPU=0 PARALLEL=5 DATASETS="hw cub"  bash scripts/ablations/run_b9_setmlp.sh
#   Terminal 2 (GPU 1): GPU=1 PARALLEL=5 DATASETS="mf"      bash scripts/ablations/run_b9_setmlp.sh
#
# Smoke:
#   SEEDS="42" DATASETS="mf" PARALLEL=1 bash scripts/ablations/run_b9_setmlp.sh
#
# DRY_RUN=1 prints commands only, does not execute.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${SEEDS:=42 43 44 45 46}"
: "${PARALLEL:=5}"
: "${GPU:=0}"
: "${DATASETS:=hw cub mf}"

# Capacity-matched hidden (overridable via env).
: "${SETMLP_HIDDEN_HW:=192}"
: "${SETMLP_HIDDEN_CUB:=128}"
: "${SETMLP_HIDDEN_MF:=144}"

n_seeds=$(echo "$SEEDS" | wc -w | tr -d ' ')
n_datasets=$(echo "$DATASETS" | wc -w | tr -d ' ')
total=$(( n_datasets * n_seeds ))

echo "B9: SetMLPFusion training (DeepSets-style, strict C2)"
echo "  datasets    : $DATASETS"
echo "  seeds       : $SEEDS"
echo "  GPU         : $GPU"
echo "  parallel    : $PARALLEL"
echo "  total runs  : $total"
echo "  hidden HW   : $SETMLP_HIDDEN_HW"
echo "  hidden CUB  : $SETMLP_HIDDEN_CUB"
echo "  hidden MF   : $SETMLP_HIDDEN_MF"

ds_base() {
  local ds="$1"
  case "$ds" in
    hw)  hw_args ;;
    cub) cub_args ;;
    mf)  mfashion_args ;;
    *)   echo "ERROR: unknown dataset '$ds'" >&2; return 1 ;;
  esac
}

ds_hidden() {
  local ds="$1"
  case "$ds" in
    hw)  echo "$SETMLP_HIDDEN_HW" ;;
    cub) echo "$SETMLP_HIDDEN_CUB" ;;
    mf)  echo "$SETMLP_HIDDEN_MF" ;;
  esac
}

# ds short tag -> actual dataset name used by main_mvc.py (already set in ds_base's --dataset;
# here it is only used for exp_name).
ds_label() {
  local ds="$1"
  case "$ds" in
    hw)  echo "hw" ;;
    cub) echo "cub" ;;
    mf)  echo "mf" ;;
  esac
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
for ds in $DATASETS; do
  base=$(ds_base "$ds")
  hidden=$(ds_hidden "$ds")
  label=$(ds_label "$ds")

  for seed in $SEEDS; do
    i=$((i+1))
    exp="B9_${label}_setmlp_h${hidden}_s${seed}"

    while (( n_running >= PARALLEL )); do
      wait -n
      n_running=$((n_running-1))
    done

    # shellcheck disable=SC2086
    run_one "$exp" $base \
      --fusion_type set_mlp \
      --set_mlp_hidden "$hidden" \
      --seed "$seed" &
    n_running=$((n_running+1))
    echo "[$i/$total] launched $exp (in_flight=$n_running)"
  done
done

echo "All $total launched. Waiting for last $n_running in flight..."
wait
echo "B9 training done."
echo ""
echo "Next: bash scripts/ablations/run_eval_b9_setmlp.sh   # 4-protocol IMVC eval"
echo ""
echo "Aggregate training:"
echo "  python aggregate_results.py --filter B9_ --out b9.csv"
