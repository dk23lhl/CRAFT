#!/usr/bin/env bash
# C9 Step 2 — Plan B 5-seed retest + Plan C (max_epochs=300) exploration.
#
# Plan B: existing C8_mf_mfth1024_long_s{42,43,44} (mean=0.9007); fill s45, s46 -> 5 seeds total.
#         Runs on GPU 0.
# Plan C: brand-new max_epochs=300 MFT x s{42,43,44} -> see whether ACC keeps rising with more epochs.
#         Runs on GPU 1.
#
# Total 5 runs; GPU 0 / GPU 1 each run 2-3 tasks in parallel, ~1-1.5h.
#
# Both means are printed automatically at the end.

set -uo pipefail

OUTPUT_ROOT=./outputs/CRAFT_MVC
LOG_DIR=./logs/ablation
mkdir -p "$LOG_DIR"

# Shared base args.
COMMON_ARGS="
  --dataset multifashion --data_dir ./data
  --encoder_type shallow --encoder_num_layers 2
  --embed_dim 256 --encoder_hidden_dim 1024
  --cluster_temperature 0.03 --noise_std 0.03 --dropout_rate 0.03
  --lambda_repr 1.0 --lambda_entropy 0.5
  --entropy_sample_weight 0.1 --entropy_warmup_epochs 10
  --kmeans_n_init 20
  --train_mask_prob 1.0 --skip_diag
  --stage1_epochs 0 --lr 1e-5
  --weight_decay 1e-4 --scheduler_T_max 100
  --output_dir ./outputs
"

run_mft() {
  local exp="$1" gpu="$2" max_ep="$3" seed="$4" resume_ckpt="$5"
  local log="$LOG_DIR/$exp.log"
  if [[ -f "$OUTPUT_ROOT/$exp/results.yaml" ]]; then
    echo "[skip] $exp"
    return 0
  fi
  if [[ ! -f "$resume_ckpt" ]]; then
    echo "[SKIP] $exp: $resume_ckpt does not exist"
    return 0
  fi
  echo "[run gpu=$gpu] $exp -> $log"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$gpu" python main_mvc.py \
    $COMMON_ARGS \
    --max_epochs "$max_ep" \
    --resume_from "$resume_ckpt" \
    --exp_name "$exp" \
    --seed "$seed" \
    > "$log" 2>&1
}

echo "C9 Step 2: Plan B (5-seed complete) + Plan C (max_ep=300) parallel"

# Plan B (GPU 0): max_epochs=200 on s45/s46.
{
  echo "[Plan B] start on GPU 0"
  for s in 45 46; do
    resume="$OUTPUT_ROOT/C7_mf_d256_h1024_s${s}/best.ckpt"
    run_mft "C8_mf_mfth1024_long_s${s}" 0 200 "$s" "$resume"
  done
  echo "[Plan B] done"
} &
PID_B=$!

# Plan C (GPU 1): max_epochs=300 on s42/s43/s44.
{
  echo "[Plan C] start on GPU 1"
  for s in 42 43 44; do
    resume="$OUTPUT_ROOT/C7_mf_d256_h1024_s${s}/best.ckpt"
    run_mft "C8_mf_mfth1024_xlong_s${s}" 1 300 "$s" "$resume"
  done
  echo "[Plan C] done"
} &
PID_C=$!

echo "Plan B PID=$PID_B (GPU 0)"
echo "Plan C PID=$PID_C (GPU 1)"
echo "Waiting both..."
wait $PID_B $PID_C
echo ""
echo "All done."
echo ""

echo "=== Plan B 5-seed mfth1024_long (max_epochs=200) ==="
accs=$(for s in 42 43 44 45 46; do
  f="$OUTPUT_ROOT/C8_mf_mfth1024_long_s${s}/results.yaml"
  [[ -f "$f" ]] && grep -E '^\s*ACC:' "$f" | head -1 | awk '{print $2}'
done)
n=$(echo "$accs" | grep -c .)
if [[ $n -gt 0 ]]; then
  mean=$(echo "$accs" | awk '{s+=$1} END{printf "%.4f", s/NR}')
  raw=$(echo "$accs" | tr '\n' ',' | sed 's/,$//')
  echo "  ${n}-seed mean = $mean  raw=[$raw]"
fi

echo ""
echo "=== Plan C 3-seed mfth1024_xlong (max_epochs=300) ==="
accs=$(for s in 42 43 44; do
  f="$OUTPUT_ROOT/C8_mf_mfth1024_xlong_s${s}/results.yaml"
  [[ -f "$f" ]] && grep -E '^\s*ACC:' "$f" | head -1 | awk '{print $2}'
done)
n=$(echo "$accs" | grep -c .)
if [[ $n -gt 0 ]]; then
  mean=$(echo "$accs" | awk '{s+=$1} END{printf "%.4f", s/NR}')
  raw=$(echo "$accs" | tr '\n' ',' | sed 's/,$//')
  echo "  ${n}-seed mean = $mean  raw=[$raw]"
fi
