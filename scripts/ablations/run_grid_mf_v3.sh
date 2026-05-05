#!/usr/bin/env bash
# C7 — multifashion architecture grid: embed_dim x encoder_hidden_dim.
#
# Goal: push multifashion complete ACC to 0.92+.
#
# Motivation: neither C2 (loss/encoder_type/entropy) nor C6 (cluster_temp/aug) grids found a
# better-than-baseline config — 5 axes are stuck. The remaining unswept high-ROI axis is model
# capacity:
#   1. embed_dim: currently 256 (inherited from HW default), unverified on mf.
#   2. encoder_hidden_dim: currently 512 (HW default); mf is small, may need different width.
#
# multifashion is 784d x 3 image data with rich classes, possibly needing a wider encoder or
# different embed_dim. Or it may need narrower/smaller (10k samples; large models may overfit).
#
# Grid (27 runs):
#   embed_dim          in {128, 256, 512}    (3)
#   encoder_hidden_dim in {256, 512, 1024}   (3)
#   seeds = 42 43 44                          (3)
#
# Naming: C7_mf_d{D}_h{H}_s{seed}
#
# Locked (from C2/C6 winners):
#   encoder_type=shallow, lambda_repr=1.0, lambda_entropy=0.5,
#   cluster_temperature=0.03, dropout_rate=0.03, noise_std=0.03
#   (mfashion_args already sets these; C6 explicitly passes lambda_repr=1.0 because the C2
#   baseline default is 1.0.)
#
# Overrides:
#   PARALLEL=4 GPUS=0    bash scripts/ablations/run_grid_mf_v3.sh
#   SEEDS="42" PARALLEL=2 bash ...                                    # smoke
#   DRY_RUN=1            bash ...

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_baseline.sh
source "$SCRIPT_DIR/_baseline.sh"

: "${SEEDS:=42 43 44}"
: "${PARALLEL:=8}"
: "${GPUS:=0,1}"

IFS=',' read -ra GPU_LIST <<< "$GPUS"
N_GPUS=${#GPU_LIST[@]}

embed_dims=(128 256 512)
hidden_dims=(256 512 1024)
n_seeds=$(echo "$SEEDS" | wc -w | tr -d ' ')
total=$(( ${#embed_dims[@]} * ${#hidden_dims[@]} * n_seeds ))

echo "C7 mfashion grid v3: embed_dim x encoder_hidden_dim"
echo "  embed_dim         : ${embed_dims[*]}"
echo "  encoder_hidden_dim: ${hidden_dims[*]}"
echo "  seeds             : $SEEDS"
echo "  parallel          : $PARALLEL across GPUs ${GPU_LIST[*]}"
echo "  total runs        : $total"

run_grid_one() {
  local exp="$1"; local gpu="$2"; shift 2
  local outdir="$OUTPUT_ROOT/$exp"
  local logfile="$LOG_DIR/$exp.log"

  if [[ -f "$outdir/results.yaml" ]]; then
    echo "    [skip] $exp"
    return 0
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "    [dry gpu=$gpu] $exp"
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

for D in "${embed_dims[@]}"; do
  for H in "${hidden_dims[@]}"; do
    for seed in $SEEDS; do
      i=$((i+1))
      exp="C7_mf_d${D}_h${H}_s${seed}"
      gpu="${GPU_LIST[$(( (i-1) % N_GPUS ))]}"

      while (( n_running >= PARALLEL )); do
        wait -n
        n_running=$((n_running-1))
      done

      # mfashion_args already has --embed_dim 256 --encoder_hidden_dim 512; passing the same
      # flag later -> argparse takes the last occurrence and overrides the default.
      # lambda_repr=1.0 is passed explicitly to match the C2 winner (mfashion_args does not set
      # lambda_repr by default).
      # shellcheck disable=SC2086
      run_grid_one "$exp" "$gpu" $base \
        --embed_dim "$D" \
        --encoder_hidden_dim "$H" \
        --lambda_repr 1.0 \
        --seed "$seed" &
      n_running=$((n_running+1))
      echo "[$i/$total] launched $exp (gpu=$gpu, in_flight=$n_running)"
    done
  done
done

echo "All $total launched. Waiting for last $n_running in flight..."
wait
echo "C7 grid done."
echo ""
echo "Aggregate:"
echo "  python aggregate_results.py --filter C7_mf_ --out c7_mf.csv"
echo "  python summarize_csv.py     --csv c7_mf.csv --out c7_mf_summary.md"
echo ""
echo "Quick top-10 by complete-data ACC:"
echo "  for d in outputs/CRAFT_MVC/C7_mf_*/; do"
echo "    acc=\$(grep -E '^\\s*ACC:' \$d/results.yaml | head -1 | awk '{print \$2}')"
echo "    printf '%s  %s\\n' \"\$acc\" \"\$(basename \$d)\""
echo "  done | sort -rn | head -10"
