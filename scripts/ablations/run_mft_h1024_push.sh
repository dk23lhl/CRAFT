#!/usr/bin/env bash
# C9 — multifashion MFT-on-h1024 final push.
#
# Based on C8 finding: mfth1024 (MFT on C7_mf_d256_h1024 ckpts) is the best (mean=0.8915,
# best=0.9256). This script does two things:
#
#   Plan A: 5-seed retest of mfth1024 -> check whether true mean is >= 0.90.
#     - Existing mfth1024 has only 3 seeds (42, 43, 44).
#     - Add s45, s46 -> 5 seeds total.
#     - But the C7 source ckpts for s45/s46 don't exist (C7 only ran 3 seeds).
#     - So first train C7_mf_d256_h1024_s{45,46}, then run MFT.
#
#   Plan B: extend MFT to max_epochs=200 (C8 used 100) x 3 seeds.
#     - Test whether "longer MFT adds more".
#     - Naming: C8_mf_mfth1024_long_s{42,43,44}
#
# Total 7 runs:
#   2 x C7 prerequisite (d256_h1024_s45, _s46)            # ~30-60min/seed
#   2 x Plan A MFT (mfth1024_s45, _s46)                    # ~10-15min
#   3 x Plan B MFT (mfth1024_long_s{42,43,44})             # ~25-30min
#
# Order: MFT_A must wait for C7 s45/s46 to finish. Plan B does not depend on new ckpts and can
# run in parallel.
#
# Total time estimate: ~1.5-2h with 4 parallel.
#
# Overrides:
#   PARALLEL=4 GPU=0 bash scripts/ablations/run_mft_h1024_push.sh
#   DRY_RUN=1 bash ...

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${PARALLEL:=4}"
: "${GPU:=0}"

base=$(mfashion_args)

run_or_skip() {
  local exp="$1"; shift
  local outdir="$OUTPUT_ROOT/$exp"
  local logfile="$LOG_DIR/$exp.log"
  if [[ -f "$outdir/results.yaml" ]]; then
    echo "    [skip] $exp"
    return 0
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "    [dry] $exp"
    return 0
  fi
  echo "    [run gpu=$GPU] $exp -> $logfile"
  CUDA_VISIBLE_DEVICES="$GPU" python "$@" > "$logfile" 2>&1
}

n_running=0
submit_bg() {
  local exp="$1"; shift
  while (( n_running >= PARALLEL )); do
    wait -n
    n_running=$((n_running-1))
  done
  run_or_skip "$exp" "$@" &
  n_running=$((n_running+1))
}

echo "C9 multifashion MFT-on-h1024 push"
echo "  GPU=$GPU  parallel=$PARALLEL"

# Step 1: train C7_mf_d256_h1024_s{45,46} (if missing).
# The 3 MFT_long jobs in Plan B can launch in parallel; they don't depend on s45/s46.
echo ""
echo "=== Step 1: prerequisites + Plan B (parallel) ==="

for s in 45 46; do
  exp="C7_mf_d256_h1024_s${s}"
  # shellcheck disable=SC2086
  submit_bg "$exp" main_mvc.py $base \
    --embed_dim 256 \
    --encoder_hidden_dim 1024 \
    --lambda_repr 1.0 \
    --exp_name "$exp" \
    --output_dir "$(dirname "$OUTPUT_ROOT")" \
    --seed "$s"
done

# Plan B: max_epochs=200 MFT on existing s42/43/44.
for s in 42 43 44; do
  resume_ckpt="$OUTPUT_ROOT/C7_mf_d256_h1024_s${s}/best.ckpt"
  if [[ ! -f "$resume_ckpt" ]]; then
    echo "  [SKIP B] s$s: $resume_ckpt does not exist"
    continue
  fi
  exp="C8_mf_mfth1024_long_s${s}"
  # shellcheck disable=SC2086
  submit_bg "$exp" main_mvc.py $base \
    --embed_dim 256 \
    --encoder_hidden_dim 1024 \
    --lambda_repr 1.0 \
    --resume_from "$resume_ckpt" \
    --stage1_epochs 0 \
    --max_epochs 200 \
    --lr 1e-5 \
    --train_mask_prob 1.0 \
    --entropy_warmup_epochs 10 \
    --skip_diag \
    --exp_name "$exp" \
    --output_dir "$(dirname "$OUTPUT_ROOT")" \
    --seed "$s"
done

echo ""
echo "Step 1 launched. Waiting for C7 prerequisites + Plan B to finish..."
wait
n_running=0
echo "Step 1 done."

# Step 2: Plan A — MFT on the freshly trained C7_mf_d256_h1024_s{45,46}.
echo ""
echo "=== Step 2: Plan A (MFT on new C7 ckpts) ==="

for s in 45 46; do
  resume_ckpt="$OUTPUT_ROOT/C7_mf_d256_h1024_s${s}/best.ckpt"
  if [[ ! -f "$resume_ckpt" ]]; then
    echo "  [SKIP A] s$s: $resume_ckpt still does not exist (C7 training failed?)"
    continue
  fi
  exp="C8_mf_mfth1024_s${s}"
  # shellcheck disable=SC2086
  submit_bg "$exp" main_mvc.py $base \
    --embed_dim 256 \
    --encoder_hidden_dim 1024 \
    --lambda_repr 1.0 \
    --resume_from "$resume_ckpt" \
    --stage1_epochs 0 \
    --max_epochs 100 \
    --lr 1e-5 \
    --train_mask_prob 1.0 \
    --entropy_warmup_epochs 10 \
    --skip_diag \
    --exp_name "$exp" \
    --output_dir "$(dirname "$OUTPUT_ROOT")" \
    --seed "$s"
done

echo ""
echo "Step 2 launched. Waiting..."
wait
echo ""
echo "All done."
echo ""
echo "=== Plan A: 5-seed mfth1024 ==="
accs=$(for s in 42 43 44 45 46; do
  f="$OUTPUT_ROOT/C8_mf_mfth1024_s${s}/results.yaml"
  [[ -f "$f" ]] && grep -E '^\s*ACC:' "$f" | head -1 | awk '{print $2}'
done)
n=$(echo "$accs" | grep -c .)
if [[ $n -gt 0 ]]; then
  mean=$(echo "$accs" | awk '{s+=$1} END{printf "%.4f", s/NR}')
  raw=$(echo "$accs" | tr '\n' ',' | sed 's/,$//')
  echo "  5-seed mean = $mean  (n=$n, raw=$raw)"
else
  echo "  no data"
fi

echo ""
echo "=== Plan B: 3-seed mfth1024_long (max_epochs=200) ==="
accs=$(for s in 42 43 44; do
  f="$OUTPUT_ROOT/C8_mf_mfth1024_long_s${s}/results.yaml"
  [[ -f "$f" ]] && grep -E '^\s*ACC:' "$f" | head -1 | awk '{print $2}'
done)
n=$(echo "$accs" | grep -c .)
if [[ $n -gt 0 ]]; then
  mean=$(echo "$accs" | awk '{s+=$1} END{printf "%.4f", s/NR}')
  raw=$(echo "$accs" | tr '\n' ',' | sed 's/,$//')
  echo "  3-seed mean = $mean  (n=$n, raw=$raw)"
else
  echo "  no data"
fi
