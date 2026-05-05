# Shared per-dataset baseline config and run helper.
# This file is **not executed directly**; it is loaded by run_*.sh via `source`.
#
# Overrides:
#   SEEDS="42 123" DATA_DIR=/data/craft bash scripts/ablations/run_b8.sh
#   DRY_RUN=1 bash scripts/ablations/run_b4.sh     # only print commands, don't execute
#
# All scripts assume execution from the **repo root**:
#   bash scripts/ablations/run_b8.sh

: "${DATA_DIR:=./data}"
: "${OUTPUT_ROOT:=./outputs/CRAFT_MVC}"
: "${LOG_DIR:=./logs/ablation}"
: "${DRY_RUN:=0}"

mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$OUTPUT_ROOT")"

# Source: paper Appendix B per-dataset config (matches paper sanity-check single run).
# HW : encoder_hidden_dim=512, embed_dim=256, lr=1e-3 (main_mvc.py default)
# CUB: encoder_hidden_dim=256 (=main_mvc.py default), embed_dim=128, lr=5e-4
#
# !!! Historical bug !!! Before 2026-04-25, cub_args incorrectly set encoder_hidden_dim=512
# and did not pass --lr explicitly. The latter was more critical: main_mvc.py default lr=1e-3,
# but paper CUB baseline actually uses 5e-4. All CUB ablation numbers were therefore low
# (~0.62 vs paper sanity baseline 0.835). HW under 1e-3 performs slightly higher than the
# paper 5e-4 baseline, no fix needed.

hw_args() {
  echo "--dataset handwritten --data_dir $DATA_DIR \
    --encoder_type deep --encoder_hidden_dim 512 \
    --embed_dim 256 --cluster_temperature 0.5 --lambda_entropy 5.0 \
    --noise_std 0.1 --dropout_rate 0.1 \
    --stage1_epochs 100 --max_epochs 200 --stage2_lr 1e-5"
}

cub_args() {
  echo "--dataset cub --data_dir $DATA_DIR \
    --encoder_type deep --encoder_hidden_dim 256 \
    --embed_dim 128 --cluster_temperature 0.5 --lambda_entropy 5.0 \
    --lr 5e-4 \
    --noise_std 0.1 --dropout_rate 0.1 \
    --stage1_epochs 100 --max_epochs 200 --stage2_lr 1e-5"
}

# multifashion: historically tuned multi-fashion config (V=3, 10k samples, 10 classes).
# Significantly different from HW/CUB: cluster_temperature=0.03 (very aggressive),
# lambda_entropy=0.5, entropy_sample_weight=0.1, training length doubled (stage1=200, max=400).
# These are multifashion-specific tuned points; grid search only sweeps lambda_repr/encoder_type/lambda_entropy.
mfashion_args() {
  echo "--dataset multifashion --data_dir $DATA_DIR \
    --encoder_type shallow --encoder_hidden_dim 512 --encoder_num_layers 2 \
    --embed_dim 256 \
    --cluster_temperature 0.03 \
    --noise_std 0.03 --dropout_rate 0.03 \
    --lambda_entropy 0.5 \
    --entropy_sample_weight 0.1 \
    --entropy_warmup_epochs 0 \
    --kmeans_n_init 20 \
    --stage1_epochs 200 --max_epochs 400 --stage2_lr 5e-5 \
    --weight_decay 1e-4 \
    --scheduler_T_max 100 \
    --lr 5e-4"
}

# Appendix datasets (paper Appx A.1 / A.2).
# Main-table ckpt naming for these 4 datasets (different from main 3 datasets C3/C4/C8):
#   Caltech    : caltech_separate_s{42-46}        — no MFT
#   UCI-digit  : uci_unified_s{42-46}             — no MFT
#   Out-Scene  : out-scene_unified_s{42-46}       — base, paired with outscene_4v_mft_s{42-46} MFT
#   YTF-31     : ytf_M_s43                        — single seed (s43, not s42 per Appx A.1)

caltech_args() {
  echo "--dataset caltech101-20 --data_dir $DATA_DIR \
    --encoder_type shallow --encoder_hidden_dim 512 --encoder_num_layers 2 \
    --embed_dim 256 --cluster_temperature 0.5 \
    --lambda_entropy 0.1 --lambda_repr 1.0 \
    --entropy_sample_weight 0.1 --entropy_warmup_epochs 15 \
    --noise_std 0.1 --dropout_rate 0.1 \
    --stage1_epochs 100 --max_epochs 200 \
    --lr 5e-4 --stage2_lr 2e-4 \
    --weight_decay 1e-4 --scheduler_T_max 100"
}

uci_args() {
  echo "--dataset uci-digit --data_dir $DATA_DIR \
    --encoder_type deep --encoder_hidden_dim 512 --encoder_num_layers 2 \
    --embed_dim 256 --cluster_temperature 0.5 \
    --lambda_entropy 5.0 --lambda_repr 1.0 \
    --entropy_sample_weight 0.1 --entropy_warmup_epochs 10 \
    --noise_std 0.1 --dropout_rate 0.1 \
    --stage1_epochs 100 --max_epochs 200 \
    --lr 5e-4 --stage2_lr 1e-5 \
    --weight_decay 1e-4 --scheduler_T_max 100"
}

# Out-Scene base (no MFT step)
outscene_args() {
  echo "--dataset out-scene --data_dir $DATA_DIR \
    --encoder_type deep --encoder_hidden_dim 512 --encoder_num_layers 2 \
    --embed_dim 256 --cluster_temperature 0.1 \
    --lambda_entropy 5.0 --lambda_repr 1.0 \
    --entropy_sample_weight 0.1 --entropy_warmup_epochs 10 \
    --noise_std 0.1 --dropout_rate 0.1 \
    --stage1_epochs 100 --max_epochs 200 \
    --lr 5e-4 --stage2_lr 1e-4 \
    --weight_decay 1e-4 --scheduler_T_max 100"
}

# YTF-31 (single seed s43; paper Appx A.1 footnote says "single run due to dataset scale n~100k")
ytf_args() {
  echo "--dataset youtubeface --data_dir $DATA_DIR \
    --encoder_type deep --encoder_hidden_dim 1024 --encoder_num_layers 2 \
    --embed_dim 512 --cluster_temperature 0.5 \
    --lambda_entropy 0.1 --lambda_repr 1.0 \
    --entropy_sample_weight 0.1 --entropy_warmup_epochs 10 \
    --noise_std 0.1 --dropout_rate 0.1 \
    --kmeans_n_init 20 \
    --stage1_epochs 100 --max_epochs 180 \
    --lr 5e-4 --stage2_lr 1e-6 \
    --weight_decay 1e-4 --scheduler_T_max 100"
}

# Idempotent run helper.
# Usage: run_if_new <exp_name> <main_mvc.py args...>
# - If results.yaml already exists -> skip (idempotent, supports resuming after interruption)
# - Each run's stdout/stderr written separately to $LOG_DIR/<exp>.log
# - DRY_RUN=1 only prints commands

run_if_new() {
  local exp_name="$1"; shift
  local outdir="$OUTPUT_ROOT/$exp_name"
  local logfile="$LOG_DIR/$exp_name.log"

  if [[ -f "$outdir/results.yaml" ]]; then
    echo "    [skip] $exp_name (results.yaml exists)"
    return 0
  fi

  local cmd=(python main_mvc.py "$@"
             --exp_name "$exp_name"
             --output_dir "$(dirname "$OUTPUT_ROOT")")

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "    [dry]  ${cmd[*]}"
    return 0
  fi

  echo "    [run]  $exp_name → $logfile"
  "${cmd[@]}" > "$logfile" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "    [FAIL] $exp_name (exit $rc); see $logfile"
    return $rc
  fi
}

count_done=0
count_skipped=0
count_failed=0
