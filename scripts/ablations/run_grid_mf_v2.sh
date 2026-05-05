#!/usr/bin/env bash
# C6 — multifashion complete-MVC push: cluster_temperature x augmentation 2D grid.
#
# Goal: bump multifashion complete ACC from 0.8823 to 0.92+.
#
# Motivation:
#   C2 grid already confirmed lambda_repr=1.0 + shallow encoder + lambda_entropy=0.5 is the
#   optimum on the loss/encoder axes. Now sweep the remaining two high-suspicion axes:
#   1. cluster_temperature: current 0.03 is a historical default (never validated); HW B6 optimum is 0.1.
#   2. augmentation (dropout=noise_std): current 0.03 is very weak; HW/CUB use 0.1.
#      multifashion is a small dataset (10k samples); weak perturbations may cause overfitting.
#
# Grid (27 runs):
#   tau in {0.03, 0.1, 0.3}    (3)  - spans one order of magnitude
#   aug in {0.03, 0.1, 0.2}    (3)  - from current mf level to HW level plus one extra step
#   seeds = 42 43 44           (3)
#
# Note: dropout_rate and noise_std change together (single aug value controls both).
#
# Naming: C6_mf_t{tau*1000}_a{aug*1000}_s{seed}
#   tau: 0.03->t30, 0.1->t100, 0.3->t300
#   aug: 0.03->a30, 0.1->a100, 0.2->a200
#
# Parallelism: 8 (4 x 2 GPUs), runs in ~3-4 hours.
#
# Overrides:
#   PARALLEL=4 GPUS=0    bash scripts/ablations/run_grid_mf_v2.sh
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

taus=(0.03 0.1 0.3)
augs=(0.03 0.1 0.2)
n_seeds=$(echo "$SEEDS" | wc -w | tr -d ' ')
total=$(( ${#taus[@]} * ${#augs[@]} * n_seeds ))

echo "C6 mfashion grid v2: tau x aug"
echo "  tau  : ${taus[*]}"
echo "  aug  : ${augs[*]} (dropout_rate=noise_std)"
echo "  seeds: $SEEDS"
echo "  parallel: $PARALLEL across GPUs ${GPU_LIST[*]}"
echo "  total runs: $total"

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

# Convert float to tag (x1000 -> integer).
to_tag() { awk "BEGIN{printf \"%d\", $1 * 1000}"; }

base=$(mfashion_args)
i=0
n_running=0

for T in "${taus[@]}"; do
  t_tag=$(to_tag "$T")
  for A in "${augs[@]}"; do
    a_tag=$(to_tag "$A")
    for seed in $SEEDS; do
      i=$((i+1))
      exp="C6_mf_t${t_tag}_a${a_tag}_s${seed}"
      gpu="${GPU_LIST[$(( (i-1) % N_GPUS ))]}"

      while (( n_running >= PARALLEL )); do
        wait -n
        n_running=$((n_running-1))
      done

      # mfashion_args already sets cluster_temperature/dropout_rate/noise_std; passing the same
      # flag later -> argparse takes the last occurrence and overrides the default.
      # Also need to preserve the C2 grid winner: lambda_repr=1.0, encoder_type=shallow,
      # lambda_entropy=0.5. mfashion_args already covers le05+shallow; only lambda_repr=1.0 is missing.
      # shellcheck disable=SC2086
      run_grid_one "$exp" "$gpu" $base \
        --cluster_temperature "$T" \
        --dropout_rate "$A" \
        --noise_std "$A" \
        --lambda_repr 1.0 \
        --seed "$seed" &
      n_running=$((n_running+1))
      echo "[$i/$total] launched $exp (gpu=$gpu, in_flight=$n_running)"
    done
  done
done

echo "All $total launched. Waiting for last $n_running in flight..."
wait
echo "C6 grid done."
echo ""
echo "Aggregate:"
echo "  python aggregate_results.py --filter C6_mf_ --out c6_mf.csv"
echo "  python summarize_csv.py     --csv c6_mf.csv --out c6_mf_summary.md"
echo ""
echo "Quick best-config peek:"
echo "  for d in outputs/CRAFT_MVC/C6_mf_*/; do"
echo "    acc=\$(grep -E '^\\s*ACC:' \$d/results.yaml | head -1 | awk '{print \$2}')"
echo "    printf '%s  %s\\n' \"\$acc\" \"\$(basename \$d)\""
echo "  done | sort -rn | head -10"
