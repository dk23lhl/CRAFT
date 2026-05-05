#!/usr/bin/env bash
# C4 — HW Mask-Aware Fine-Tuning (MFT) on top of C3_best_hw ckpts.
#
# Motivation: C3 trains entirely on complete data, so mask_token never appears in forward ->
# at inference it is a random noise vector (the C3_best_hw_s46 diagnostic shows mask_token
# L2/sqrt(numel)=0.019 ~= init std 0.02, confirming it is dead).
# MFT continues training with random view masking, letting mask_token learn -> inference can
# actually use it.
#
# Total 5 runs: 5 seeds x 1 dataset (HW).
#
# Key parameters:
#   --resume_from C3_best_hw_s{seed}/best.ckpt   # load weights, reset optimizer/epoch
#   --stage1_epochs 0                             # skip pretraining (already done)
#   --max_epochs 50 --lr 1e-5                     # short + small lr fine-tuning
#   --train_mask_prob 0.5                         # 50% probability of random mask per batch
#   --skip_diag                                   # skip random-encoder diagnostic
#   Keep all hw_args architecture + C3 overrides (lambda_repr=0.5, tau=0.1); otherwise model
#   build will not match.
#
# Naming: C4_hw_mft_p{P}_s{seed}, P=05 for train_mask_prob=0.5.
#
# Overrides:
#   TRAIN_MASK_PROB=0.3 bash scripts/ablations/run_mft_hw.sh
#   MFT_EPOCHS=100 MFT_LR=5e-5 bash ...
#   GPU=1 PARALLEL=3 bash ...
#   DRY_RUN=1 bash ...

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${SEEDS:=42 43 44 45 46}"
: "${PARALLEL:=2}"
: "${GPU:=0}"

# MFT-specific defaults — aligned with the previously validated, well-performing MFT recipe
# (the hw_6v_mft_s* batch):
#   train_mask_prob=1.0  : mask every batch (not 0.5), so mask_token gets sufficient gradient
#   max_epochs=100       : previously used 100; trains more stably than 50
#   lr=1e-5              : small lr fine-tuning
: "${TRAIN_MASK_PROB:=1.0}"
: "${MFT_EPOCHS:=100}"
: "${MFT_LR:=1e-5}"

# Loss/regularization overrides — aligned with the previous MFT recipe:
#   lambda_repr=1.0          : prior MFT used 1.0 (not the 0.5 from C3 training)
#   cluster_temperature=0.1  : same as C3
#   entropy_sample_weight=0.1: prior MFT used 0.1 (weaken sample-entropy)
#   entropy_warmup_epochs=10 : warmup the entropy term for 10 epochs
: "${HW_OVERRIDE:=--lambda_repr 1.0 --cluster_temperature 0.1 --entropy_sample_weight 0.1 --entropy_warmup_epochs 10}"

: "${SOURCE_PREFIX:=C3_best_hw}"

mft_tag=$(echo "$TRAIN_MASK_PROB" | tr -d '.')
n_seeds=$(echo "$SEEDS" | wc -w | tr -d ' ')

echo "C4: HW Mask-Aware Fine-Tuning"
echo "  source ckpts : ${SOURCE_PREFIX}_s{${SEEDS// /,}}/best.ckpt"
echo "  HW overrides : $HW_OVERRIDE"
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
hw_base=$(hw_args)

for seed in $SEEDS; do
  i=$((i+1))
  resume_ckpt="$OUTPUT_ROOT/${SOURCE_PREFIX}_s${seed}/best.ckpt"
  if [[ ! -f "$resume_ckpt" ]]; then
    echo "[$i/$n_seeds] [SKIP] $resume_ckpt does not exist"
    continue
  fi
  exp="C4_hw_mft_p${mft_tag}_s${seed}"

  while (( n_running >= PARALLEL )); do
    wait -n
    n_running=$((n_running-1))
  done

  # shellcheck disable=SC2086
  run_one "$exp" "$resume_ckpt" \
    $hw_base $HW_OVERRIDE \
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
echo "Next: bash scripts/ablations/run_eval_mft_hw.sh"
echo ""
echo "Aggregate training:"
echo "  python aggregate_results.py --filter C4_hw_mft_ --out c4_mft.csv"
