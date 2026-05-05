#!/usr/bin/env bash
# C8 — multifashion complete-data last-attempt comprehensive grid.
#
# Combines all unswept axes + MFT variants. Every config is based on mfashion_args (C2/C6/C7
# winner) and changes only one axis. If any config can break 0.92, it must show up here.
#
# 36 runs, 6-8 parallel ~5-7h; run with nohup overnight.
# Fully idempotent: skips existing results.yaml automatically.
#
# Covers 7 phases:
#   Phase 1 [9 runs]: lambda_recon in {0.5, 2.0, 5.0} x 3 seeds
#   Phase 2 [3 runs]: repr_mode = vicreg x 3 seeds
#   Phase 3 [6 runs]: fusion_n_layers in {2, 3} x 3 seeds
#   Phase 4 [6 runs]: encoder_num_layers in {3, 4} x 3 seeds (shallow)
#   Phase 5 [6 runs]: max_epochs in {600, 800} x 3 seeds
#   Phase 6 [3 runs]: MFT-A train_mask_prob=0.3 + max_epochs=200 (regularization angle)
#   Phase 7 [3 runs]: MFT-B on top of C7_mf_d256_h1024 lucky ckpts (stabilize lucky shot)
#
# Naming: C8_mf_<tag>_s<seed>
#   lrec05/lrec20/lrec50 (lambda_recon x 10)
#   vicreg
#   fl2/fl3 (fusion_n_layers)
#   el3/el4 (encoder_num_layers)
#   me600/me800 (max_epochs)
#   mftp03 (MFT-A, train_mask_prob=0.3)
#   mfth1024 (MFT-B, h1024 source)
#
# Aggregate after running:
#   python aggregate_results.py --filter C8_mf_ --out c8_mf.csv
#   python summarize_csv.py     --csv c8_mf.csv --out c8_mf_summary.md
#
# Overrides:
#   PARALLEL=4 GPUS=0   bash scripts/ablations/run_grid_mf_v4_final.sh
#   SEEDS="42" PARALLEL=2 bash ...                                       # smoke
#   DRY_RUN=1           bash ...

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${SEEDS:=42 43 44}"
: "${PARALLEL:=8}"
: "${GPUS:=0,1}"

IFS=',' read -ra GPU_LIST <<< "$GPUS"
N_GPUS=${#GPU_LIST[@]}

base=$(mfashion_args)

i=0
n_running=0
gpu_idx=0

# Generic submitter: submit "exp_name" "python_arg1 arg2 ..."
# python_args is a single string containing all flags for main_mvc.py or incomplete_eval.py.
submit() {
  local exp="$1"; shift
  local outdir="$OUTPUT_ROOT/$exp"
  local logfile="$LOG_DIR/$exp.log"

  while (( n_running >= PARALLEL )); do
    wait -n
    n_running=$((n_running-1))
  done

  i=$((i+1))

  if [[ -f "$outdir/results.yaml" ]]; then
    echo "[$i] [skip] $exp"
    return 0
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[$i] [dry] $exp"
    return 0
  fi

  local gpu="${GPU_LIST[$(( gpu_idx % N_GPUS ))]}"
  gpu_idx=$((gpu_idx + 1))

  echo "[$i] [run gpu=$gpu] $exp -> $logfile"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$gpu" python "$@" > "$logfile" 2>&1 &
  n_running=$((n_running+1))
}

echo "C8 mfashion last-attempt grid"
echo "  seeds      : $SEEDS"
echo "  parallel   : $PARALLEL across GPUs ${GPU_LIST[*]}"
echo "  expected   : 36 runs"

# Phase 1: lambda_recon scan (9 runs).
# multifashion has image views; reconstruction weight may want to be different from 1.0.
echo ""
echo "=== Phase 1: lambda_recon in {0.5, 2.0, 5.0} ==="
for L in 0.5 2.0 5.0; do
  l_tag=$(awk "BEGIN{printf \"%d\", $L * 10}")   # 0.5->05, 2.0->20, 5.0->50
  for s in $SEEDS; do
    exp="C8_mf_lrec${l_tag}_s${s}"
    # shellcheck disable=SC2086
    submit "$exp" main_mvc.py $base \
      --lambda_recon "$L" \
      --lambda_repr 1.0 \
      --exp_name "$exp" \
      --output_dir "$(dirname "$OUTPUT_ROOT")" \
      --seed "$s"
  done
done

# Phase 2: repr_mode = vicreg (3 runs).
# A completely different alignment loss; may open a new ceiling.
echo ""
echo "=== Phase 2: repr_mode=vicreg ==="
for s in $SEEDS; do
  exp="C8_mf_vicreg_s${s}"
  # shellcheck disable=SC2086
  submit "$exp" main_mvc.py $base \
    --repr_mode vicreg \
    --lambda_repr 1.0 \
    --exp_name "$exp" \
    --output_dir "$(dirname "$OUTPUT_ROOT")" \
    --seed "$s"
done

# Phase 3: fusion_n_layers in {2, 3} (6 runs).
# B1 finds L=1 optimal on HW, but unverified on mf; more fusion layers may capture more complex
# view interactions.
echo ""
echo "=== Phase 3: fusion_n_layers in {2, 3} ==="
for L in 2 3; do
  for s in $SEEDS; do
    exp="C8_mf_fl${L}_s${s}"
    # shellcheck disable=SC2086
    submit "$exp" main_mvc.py $base \
      --fusion_n_layers "$L" \
      --lambda_repr 1.0 \
      --exp_name "$exp" \
      --output_dir "$(dirname "$OUTPUT_ROOT")" \
      --seed "$s"
  done
done

# Phase 4: encoder_num_layers in {3, 4} with encoder_type=shallow (6 runs).
# C2 confirmed deep type (4 layer 500-500-2000) doesn't work, but deepening shallow may differ.
echo ""
echo "=== Phase 4: encoder_num_layers in {3, 4} (shallow) ==="
for E in 3 4; do
  for s in $SEEDS; do
    exp="C8_mf_el${E}_s${s}"
    # shellcheck disable=SC2086
    submit "$exp" main_mvc.py $base \
      --encoder_num_layers "$E" \
      --lambda_repr 1.0 \
      --exp_name "$exp" \
      --output_dir "$(dirname "$OUTPUT_ROOT")" \
      --seed "$s"
  done
done

# Phase 5: max_epochs in {600, 800} (6 runs).
# Currently stage1=200, max=400 (stage2=200). Extending may let stage2 fully converge.
echo ""
echo "=== Phase 5: max_epochs in {600, 800} ==="
for M in 600 800; do
  for s in $SEEDS; do
    exp="C8_mf_me${M}_s${s}"
    # shellcheck disable=SC2086
    submit "$exp" main_mvc.py $base \
      --max_epochs "$M" \
      --lambda_repr 1.0 \
      --exp_name "$exp" \
      --output_dir "$(dirname "$OUTPUT_ROOT")" \
      --seed "$s"
  done
done

# Phase 6: MFT-A (light mask + longer training, regularization angle) (3 runs).
# train_mask_prob=0.3 (vs C5's 1.0), so 70% of batches still see complete data;
# max_epochs=200 (vs C5's 100). Hope: mask augmentation as a gentle regularizer.
echo ""
echo "=== Phase 6: MFT-A (train_mask_prob=0.3, longer) ==="
for s in $SEEDS; do
  resume_ckpt="$OUTPUT_ROOT/C2_mf_lr10_s_le05_s${s}/best.ckpt"
  if [[ ! -f "$resume_ckpt" ]]; then
    echo "  [SKIP] s$s: $resume_ckpt does not exist"
    continue
  fi
  exp="C8_mf_mftp03_s${s}"
  # shellcheck disable=SC2086
  submit "$exp" main_mvc.py $base \
    --resume_from "$resume_ckpt" \
    --stage1_epochs 0 \
    --max_epochs 200 \
    --lr 1e-5 \
    --train_mask_prob 0.3 \
    --entropy_warmup_epochs 10 \
    --lambda_repr 1.0 \
    --skip_diag \
    --exp_name "$exp" \
    --output_dir "$(dirname "$OUTPUT_ROOT")" \
    --seed "$s"
done

# Phase 7: MFT-B (MFT on top of C7 lucky h1024 ckpts) (3 runs).
# C7_mf_d256_h1024 single seed reached 0.9085, but 3-seed mean dropped to 0.8798.
# Try MFT to stabilize this lucky state without destroying it.
echo ""
echo "=== Phase 7: MFT-B (on C7_mf_d256_h1024 ckpts) ==="
for s in $SEEDS; do
  resume_ckpt="$OUTPUT_ROOT/C7_mf_d256_h1024_s${s}/best.ckpt"
  if [[ ! -f "$resume_ckpt" ]]; then
    echo "  [SKIP] s$s: $resume_ckpt does not exist"
    continue
  fi
  exp="C8_mf_mfth1024_s${s}"
  # MFT-B must use C7's architecture (d=256, h=1024).
  # shellcheck disable=SC2086
  submit "$exp" main_mvc.py $base \
    --embed_dim 256 \
    --encoder_hidden_dim 1024 \
    --resume_from "$resume_ckpt" \
    --stage1_epochs 0 \
    --max_epochs 100 \
    --lr 1e-5 \
    --train_mask_prob 1.0 \
    --entropy_warmup_epochs 10 \
    --lambda_repr 1.0 \
    --skip_diag \
    --exp_name "$exp" \
    --output_dir "$(dirname "$OUTPUT_ROOT")" \
    --seed "$s"
done

echo ""
echo "All $i jobs launched. Waiting for last $n_running in flight..."
wait
echo ""
echo "C8 grid done."
echo ""
echo "Top-15 by complete-data ACC:"
for d in "$OUTPUT_ROOT"/C8_mf_*/; do
  [[ -f "$d/results.yaml" ]] || continue
  acc=$(grep -E '^\s*ACC:' "$d/results.yaml" | head -1 | awk '{print $2}')
  printf '%s  %s\n' "$acc" "$(basename "$d")"
done | sort -rn | head -15
echo ""
echo "Aggregate:"
echo "  python aggregate_results.py --filter C8_mf_ --out c8_mf.csv"
echo "  python summarize_csv.py     --csv c8_mf.csv --out c8_mf_summary.md"
