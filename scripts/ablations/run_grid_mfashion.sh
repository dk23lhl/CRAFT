#!/usr/bin/env bash
# C2 — multifashion full grid search.
#
# Grid (60 runs):
#   lambda_repr    in {0.0, 0.1, 0.3, 0.5, 1.0}    (5)
#   encoder_type   in {shallow, deep}              (2)
#   lambda_entropy in {0.5, 5.0}                   (2)
#   seeds          = 42 43 44                      (3)
# Total = 5 x 2 x 2 x 3 = 60 runs.
#
# Motivation: B5 finds lambda_repr=0.1 optimal on CUB (V=2) (U-shaped curve, sweet spot at a
# small non-zero value). multifashion (V=3) lies between CUB and HW; need a sweep to find the
# optimum. Stack two secondary axes: encoder_type (CUB benefits from deep) and lambda_entropy
# (multifashion historically uses 0.5 vs HW/CUB's 5.0). Remaining hyperparameters use the tuned
# multifashion config (mfashion_args).
#
# Parallelism: default PARALLEL=8 (4 procs/GPU x 2 GPUs); 24G VRAM is comfortable.
#       wait -n maintains concurrency limit; idempotent (skip if results.yaml exists).
#
# Naming: C2_mf_lr{LR}_{enc}_le{LE}_s{seed}
#   LR : lambda_repr x 10  (0.0->00, 0.1->01, 0.3->03, 0.5->05, 1.0->10)
#   enc: s = shallow, d = deep
#   LE : lambda_entropy x 10 (0.5->05, 5.0->50)
#
# Overrides:
#   PARALLEL=4 GPUS=0   bash scripts/ablations/run_grid_mfashion.sh   # single GPU, 4 parallel
#   PARALLEL=2 SEEDS="42" bash scripts/ablations/run_grid_mfashion.sh # smoke
#   DRY_RUN=1            bash scripts/ablations/run_grid_mfashion.sh

# Do not use set -e: a single failed run should not interrupt the batch (the other 59 are still running).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${SEEDS:=42 43 44}"
: "${PARALLEL:=8}"
: "${GPUS:=0,1}"

IFS=',' read -ra GPU_LIST <<< "$GPUS"
N_GPUS=${#GPU_LIST[@]}

lambdas=(0.0 0.1 0.3 0.5 1.0)
encoders=(shallow deep)
entropies=(0.5 5.0)
n_seeds=$(echo "$SEEDS" | wc -w | tr -d ' ')
total=$(( ${#lambdas[@]} * ${#encoders[@]} * ${#entropies[@]} * n_seeds ))

echo "C2 mfashion grid search"
echo "  lambda_repr   : ${lambdas[*]}"
echo "  encoder       : ${encoders[*]}"
echo "  lambda_entropy: ${entropies[*]}"
echo "  seeds         : $SEEDS"
echo "  parallel      : $PARALLEL across GPUs ${GPU_LIST[*]}"
echo "  total runs    : $total"

# Single-run worker, called in a subshell in parallel (launched with &).
run_grid_one() {
  local exp="$1"; local gpu="$2"; shift 2
  local outdir="$OUTPUT_ROOT/$exp"
  local logfile="$LOG_DIR/$exp.log"

  if [[ -f "$outdir/results.yaml" ]]; then
    echo "    [skip] $exp (results exists)"
    return 0
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "    [dry gpu=$gpu] $exp"
    echo "      python main_mvc.py $* --exp_name $exp" \
         "--output_dir $(dirname "$OUTPUT_ROOT")" >> "$LOG_DIR/_grid_dry.log"
    return 0
  fi
  echo "    [run gpu=$gpu] $exp -> $logfile"
  CUDA_VISIBLE_DEVICES="$gpu" python main_mvc.py "$@" \
    --exp_name "$exp" \
    --output_dir "$(dirname "$OUTPUT_ROOT")" \
    > "$logfile" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "    [FAIL] $exp (exit $rc); see $logfile"
  fi
}

base=$(mfashion_args)
i=0
n_running=0
[[ "${DRY_RUN:-0}" == "1" ]] && : > "$LOG_DIR/_grid_dry.log"

for L in "${lambdas[@]}"; do
  lr_tag=$(echo "$L" | tr -d '.')
  for E in "${encoders[@]}"; do
    enc_tag="${E:0:1}"
    for LE in "${entropies[@]}"; do
      le_tag=$(echo "$LE" | tr -d '.')
      for seed in $SEEDS; do
        i=$((i+1))
        exp="C2_mf_lr${lr_tag}_${enc_tag}_le${le_tag}_s${seed}"
        gpu="${GPU_LIST[$(( (i-1) % N_GPUS ))]}"

        while (( n_running >= PARALLEL )); do
          wait -n
          n_running=$((n_running-1))
        done

        # mfashion_args already includes --encoder_type/--lambda_entropy defaults; passing the
        # same flag later -> argparse takes the last occurrence and overrides the default.
        # shellcheck disable=SC2086
        run_grid_one "$exp" "$gpu" $base \
          --encoder_type "$E" \
          --lambda_repr "$L" \
          --lambda_entropy "$LE" \
          --seed "$seed" &
        n_running=$((n_running+1))
        echo "[$i/$total] launched $exp (gpu=$gpu, in_flight=$n_running)"
      done
    done
  done
done

echo "All $total launched. Waiting for $n_running in flight..."
wait
echo "Grid done."
echo ""
echo "Aggregate:"
echo "  python aggregate_results.py --filter C2_mf_ --out c2_mf.csv"
echo "  python summarize_csv.py --csv c2_mf.csv --out c2_mf_summary.md"
echo ""
echo "Quick best-config peek (top 5 by complete-data ACC):"
echo "  for d in outputs/CRAFT_MVC/C2_mf_*/; do"
echo "    acc=\$(grep -E '^\\s*ACC:' \$d/results.yaml | head -1 | awk '{print \$2}')"
echo "    printf '%s  %s\\n' \"\$acc\" \"\$(basename \$d)\""
echo "  done | sort -rn | head -5"
