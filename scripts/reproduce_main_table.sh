#!/usr/bin/env bash
# Reproduce paper Table 2 (main, V=2/3/6) + Tables 10-13 (appendix datasets).
#
# 7 datasets × up to 5 seeds × (Stage 1+2 + optional MFT) + 4-protocol IMVC eval.
# Total wall-clock estimate (single RTX 3090): ~12-18 hours.
#
# All ckpts written to ./outputs/CRAFT_MVC/<exp_name>/best.ckpt; idempotent — already
# trained ckpts are skipped on re-run. Per-(P, r) IMVC eval results are written to
# <exp_name>/four_protocol_eval_results.yaml; aggregated into main_table.csv at the end.
#
# Usage:
#   bash scripts/reproduce_main_table.sh
#   GPU=1 PARALLEL=4 bash scripts/reproduce_main_table.sh
#   SEEDS="42" DRY_RUN=1 bash scripts/reproduce_main_table.sh   # smoke
#
# Skip individual phases:
#   SKIP_TRAIN=1   bash ...   # only eval + aggregate
#   SKIP_EVAL=1    bash ...   # only train

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ablations/_baseline.sh
source "$SCRIPT_DIR/ablations/_baseline.sh"

: "${SEEDS:=42 43 44 45 46}"
: "${PARALLEL:=4}"
: "${GPU:=0}"
: "${SKIP_TRAIN:=0}"
: "${SKIP_EVAL:=0}"

n_running=0

submit_bg() {
  local exp="$1"; shift
  local outdir="$OUTPUT_ROOT/$exp"
  local logfile="$LOG_DIR/$exp.log"
  if [[ -f "$outdir/results.yaml" ]]; then
    echo "    [skip-train] $exp"
    return 0
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "    [dry] $exp"
    return 0
  fi
  while (( n_running >= PARALLEL )); do
    wait -n
    n_running=$((n_running-1))
  done
  echo "    [run gpu=$GPU] $exp -> $logfile"
  CUDA_VISIBLE_DEVICES="$GPU" python main_mvc.py "$@" \
    --exp_name "$exp" --output_dir "$(dirname "$OUTPUT_ROOT")" \
    > "$logfile" 2>&1 &
  n_running=$((n_running+1))
}

eval_one() {
  local exp="$1"; local dataset="$2"
  local ckpt="$OUTPUT_ROOT/$exp/best.ckpt"
  local out="$OUTPUT_ROOT/$exp/four_protocol_eval_results.yaml"
  local log="$LOG_DIR/eval_$exp.log"
  [[ ! -f "$ckpt" ]] && { echo "    [skip-eval] $exp (no ckpt)"; return 0; }
  [[ -f "$out" ]] && { echo "    [skip-eval] $exp (results exist)"; return 0; }
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "    [dry-eval] $exp ($dataset)"
    return 0
  fi
  while (( n_running >= PARALLEL )); do
    wait -n
    n_running=$((n_running-1))
  done
  echo "    [eval gpu=$GPU] $exp ($dataset)"
  CUDA_VISIBLE_DEVICES="$GPU" python incomplete_eval.py \
    --checkpoint "$ckpt" --dataset "$dataset" --data_dir "$DATA_DIR" \
    --protocol all --missing_rates 0.1 0.3 0.5 0.7 \
    --num_trials 5 --seed 42 \
    > "$log" 2>&1 &
  n_running=$((n_running+1))
}

echo "========================================"
echo "Reproduce paper main table (Tables 2, 10, 11, 12, 13)"
echo "  seeds      : $SEEDS"
echo "  GPU        : $GPU"
echo "  parallel   : $PARALLEL"
echo "  skip train : $SKIP_TRAIN"
echo "  skip eval  : $SKIP_EVAL"
echo "========================================"

# =========================================================================
# Phase 1: Train base ckpts (Stage 1 + Stage 2, no MFT)
# =========================================================================
if [[ "$SKIP_TRAIN" != "1" ]]; then
  echo ""
  echo "=== Phase 1: Train base CRAFT (no MFT) ==="

  hw_base=$(hw_args)
  cub_base=$(cub_args)
  mf_base=$(mfashion_args)
  cal_base=$(caltech_args)
  uci_base=$(uci_args)
  os_base=$(outscene_args)
  ytf_base=$(ytf_args)

  for s in $SEEDS; do
    # HW (V=6) base — used as resume_from for HW MFT
    # shellcheck disable=SC2086
    submit_bg "C3_best_hw_s${s}" $hw_base --lambda_repr 0.5 --cluster_temperature 0.1 --seed "$s"

    # CUB (V=2) — main-table CRAFT directly (no MFT for V=2 per §5.3)
    # shellcheck disable=SC2086
    submit_bg "C3_best_cub_s${s}" $cub_base --lambda_repr 0.1 --seed "$s"

    # MF (V=3) base = C7_mf_d256_h1024 — used as resume_from for MF MFT (xlong)
    # shellcheck disable=SC2086
    submit_bg "C7_mf_d256_h1024_s${s}" $mf_base --encoder_hidden_dim 1024 --lambda_repr 1.0 --seed "$s"

    # Caltech (V=6) — no MFT
    # shellcheck disable=SC2086
    submit_bg "caltech_separate_s${s}" $cal_base --seed "$s"

    # UCI-digit (V=3) — no MFT
    # shellcheck disable=SC2086
    submit_bg "uci_unified_s${s}" $uci_base --seed "$s"

    # Out-Scene (V=4) base — used as resume_from for OS MFT
    # shellcheck disable=SC2086
    submit_bg "out-scene_unified_s${s}" $os_base --seed "$s"
  done

  # YTF-31 (V=5): single seed s43 only (paper Appx A.1, dataset scale n≈100k)
  # shellcheck disable=SC2086
  submit_bg "ytf_M_s43" $ytf_base --kmeans_n_init 20 --seed 43

  echo "    Phase 1 launched. Waiting..."
  wait
  n_running=0
  echo "    Phase 1 done."

  # =======================================================================
  # Phase 2: MFT step (HW, MF, Out-Scene; V≥3 datasets that benefit per §5.3)
  # =======================================================================
  echo ""
  echo "=== Phase 2: MFT fine-tuning on top of Phase 1 ckpts ==="

  for s in $SEEDS; do
    # HW MFT: 100 ep at lr=1e-5, train_mask_prob=1.0, lambda_repr override 0.5→1.0
    resume="$OUTPUT_ROOT/C3_best_hw_s${s}/best.ckpt"
    if [[ -f "$resume" ]]; then
      # shellcheck disable=SC2086
      submit_bg "C4_hw_mft_p10_s${s}" $hw_base \
        --lambda_repr 1.0 --cluster_temperature 0.1 \
        --resume_from "$resume" \
        --stage1_epochs 0 --max_epochs 100 --lr 1e-5 \
        --train_mask_prob 1.0 --skip_diag --seed "$s"
    fi

    # MF MFT (xlong): 300 ep at lr=1e-5
    resume="$OUTPUT_ROOT/C7_mf_d256_h1024_s${s}/best.ckpt"
    if [[ -f "$resume" ]]; then
      # shellcheck disable=SC2086
      submit_bg "C8_mf_mfth1024_xlong_s${s}" $mf_base \
        --encoder_hidden_dim 1024 --lambda_repr 1.0 \
        --resume_from "$resume" \
        --stage1_epochs 0 --max_epochs 300 --lr 1e-5 \
        --train_mask_prob 1.0 --skip_diag --seed "$s"
    fi

    # Out-Scene MFT: 100 ep at lr=1e-5
    resume="$OUTPUT_ROOT/out-scene_unified_s${s}/best.ckpt"
    if [[ -f "$resume" ]]; then
      # shellcheck disable=SC2086
      submit_bg "outscene_4v_mft_s${s}" $os_base \
        --resume_from "$resume" \
        --stage1_epochs 0 --max_epochs 100 --lr 1e-5 \
        --train_mask_prob 1.0 --skip_diag --seed "$s"
    fi
  done

  echo "    Phase 2 launched. Waiting..."
  wait
  n_running=0
  echo "    Phase 2 done."
fi

# =========================================================================
# Phase 3: IMVC eval (all 4 protocols × 4 missing rates) for each main-table ckpt
# =========================================================================
if [[ "$SKIP_EVAL" != "1" ]]; then
  echo ""
  echo "=== Phase 3: 4-protocol IMVC evaluation ==="

  for s in $SEEDS; do
    eval_one "C4_hw_mft_p10_s${s}"        handwritten
    eval_one "C3_best_cub_s${s}"          cub
    eval_one "C8_mf_mfth1024_xlong_s${s}" multifashion
    eval_one "caltech_separate_s${s}"     caltech101-20
    eval_one "uci_unified_s${s}"          uci-digit
    eval_one "outscene_4v_mft_s${s}"      out-scene
  done
  eval_one "ytf_M_s43" youtubeface

  echo "    Phase 3 launched. Waiting..."
  wait
  n_running=0
  echo "    Phase 3 done."
fi

# =========================================================================
# Phase 4: Aggregate into a single CSV + markdown
# =========================================================================
echo ""
echo "=== Phase 4: Aggregate ==="
python aggregate_imvc_results.py \
  --filter "C4_hw_mft_p10_,C3_best_cub_,C8_mf_mfth1024_xlong_,caltech_separate_,uci_unified_,outscene_4v_mft_,ytf_M_" \
  --out main_table.csv
python summarize_csv.py --csv main_table.csv --out main_table_summary.md

echo "========================================"
echo "Done. Outputs:"
echo "  main_table.csv         (raw seeds × cells)"
echo "  main_table_summary.md  (5-seed mean ± std vs paper Table 2/10-13)"
echo "========================================"
