#!/usr/bin/env bash
# C5 — multifashion Mask-Aware Fine-Tuning on top of C2_mf_lr10_s_le05 ckpts.
#
# Motivation: C2 grid confirms lambda_repr=1.0 + shallow + lambda_entropy=0.5 is optimal on
# multifashion (0.8823 +/- 0.011), but mask_token has never been trained under IMVC (missing views)
# (same problem HW had previously). Use the MFT recipe validated on HW (train_mask_prob=1.0,
# max_epochs=100) to let mask_token learn.
#
# Total 3 runs: 3 seeds x 1 dataset (multifashion).
# Note: the multifashion C2 grid defaults to seeds 42/43/44 only. To get 5 seeds, extend C2 grid first, then MFT.
#
# Key parameters:
#   --resume_from C2_mf_lr10_s_le05_s{seed}/best.ckpt
#   --stage1_epochs 0
#   --max_epochs 100 --lr 1e-5
#   --train_mask_prob 1.0     (validated on HW to be better than 0.5)
#   --skip_diag
#   Keep all mfashion_args + lambda_repr=1.0 (C2 winner) + entropy_warmup_epochs=10 (HW MFT recipe).
#
# Naming: C5_mf_mft_p{P}_s{seed}, P=10 for train_mask_prob=1.0.
#
# Overrides:
#   TRAIN_MASK_PROB=0.7 bash scripts/ablations/run_mft_mf.sh
#   MFT_EPOCHS=50 bash ...
#   GPU=1 PARALLEL=3 bash ...
#   DRY_RUN=1 bash ...

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${SEEDS:=42 43 44}"
: "${PARALLEL:=2}"
: "${GPU:=0}"

# MFT-specific defaults — aligned with the C4 v2 recipe validated on HW.
: "${TRAIN_MASK_PROB:=1.0}"
: "${MFT_EPOCHS:=100}"
: "${MFT_LR:=1e-5}"

# Loss/regularization overrides:
#   lambda_repr=1.0           : C2 grid optimum on multifashion (different from CUB)
#   entropy_warmup_epochs=10  : reuse HW MFT recipe to avoid early entropy collapse
#   Other mfashion-specific parameters (cluster_temp=0.03, lambda_entropy=0.5,
#     entropy_sample_weight=0.1, shallow encoder, embed_dim=256, etc.) come from mfashion_args.
: "${MF_OVERRIDE:=--lambda_repr 1.0 --entropy_warmup_epochs 10}"

: "${SOURCE_PREFIX:=C2_mf_lr10_s_le05}"

mft_tag=$(echo "$TRAIN_MASK_PROB" | tr -d '.')
n_seeds=$(echo "$SEEDS" | wc -w | tr -d ' ')

echo "C5: multifashion Mask-Aware Fine-Tuning"
echo "  source ckpts : ${SOURCE_PREFIX}_s{${SEEDS// /,}}/best.ckpt"
echo "  MF overrides : $MF_OVERRIDE"
echo "  train_mask_p : $TRAIN_MASK_PROB"
echo "  max_epochs   : $MFT_EPOCHS"
echo "  lr           : $MFT_LR"
echo "  GPU          : $GPU"
echo "  parallel     : $PARALLEL"
echo "  total runs   : $n_seeds"

run_one() {
  local exp="$1"; local resume_ckpt="$2"; shift 2
  local outdir="$OUTPUT_ROOT/$exp"
  local logfile="$LOG_DIR/$exp.log"

  if [[ -f "$outdir/results.yaml" ]]; then
    echo "    [skip] $exp (results exists)"
    return 0
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "    [dry gpu=$GPU] $exp <- $resume_ckpt"
    return 0
  fi
  echo "    [run gpu=$GPU] $exp <- $(basename $(dirname $resume_ckpt))"
  CUDA_VISIBLE_DEVICES="$GPU" python main_mvc.py "$@" \
    --resume_from "$resume_ckpt" \
    --exp_name "$exp" \
    --output_dir "$(dirname "$OUTPUT_ROOT")" \
    > "$logfile" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "    [FAIL] $exp (exit $rc); see $logfile"
  fi
}

i=0
n_running=0
mf_base=$(mfashion_args)

for seed in $SEEDS; do
  i=$((i+1))
  resume_ckpt="$OUTPUT_ROOT/${SOURCE_PREFIX}_s${seed}/best.ckpt"
  if [[ ! -f "$resume_ckpt" ]]; then
    echo "[$i/$n_seeds] [SKIP] $resume_ckpt does not exist"
    continue
  fi
  exp="C5_mf_mft_p${mft_tag}_s${seed}"

  while (( n_running >= PARALLEL )); do
    wait -n
    n_running=$((n_running-1))
  done

  # shellcheck disable=SC2086
  run_one "$exp" "$resume_ckpt" \
    $mf_base $MF_OVERRIDE \
    --stage1_epochs 0 \
    --max_epochs "$MFT_EPOCHS" \
    --lr "$MFT_LR" \
    --train_mask_prob "$TRAIN_MASK_PROB" \
    --skip_diag \
    --seed "$seed" &
  n_running=$((n_running+1))
  echo "[$i/$n_seeds] launched $exp (in_flight=$n_running)"
done

echo "All $n_seeds launched. Waiting for $n_running in flight..."
wait
echo "MFT training done."
echo ""
echo "Next: bash scripts/ablations/run_eval_mft_mf.sh"
echo ""
echo "Aggregate training:"
echo "  python aggregate_results.py --filter C5_mf_mft_ --out c5_mft.csv"
