#!/usr/bin/env bash
# Reproduce paper Table 16 (Appx D.9): trainability isolation cells on HandWritten.
#
# 3 cells (A/C/D) are sliced from the existing C4_hw_mft_p10_s{42-46} IMVC eval
# results — NO additional training required.
#
#   Cell A: Protocol 1 at r=0.7      r̂≈11.7%, p_c≈30%
#   Cell C: Protocol 1 at r=1.0      r̂≈16.67%, p_c=0  (5-seed mean)
#   Cell D: Protocol 4 at r=0.7      r̂=70%,    p_c≈0
#
# Cell C requires a separate eval run (rate=1.0 not in default eval grid).
# CRAFT-Core (Stage 1 only) numbers come from Core_hw_s{42-46} ckpts; reproduce
# those via scripts/reproduce_craft_core.sh first.
#
# Usage:
#   bash scripts/reproduce_isolation_cell.sh
#   GPU=1 bash scripts/reproduce_isolation_cell.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ablations/_baseline.sh
source "$SCRIPT_DIR/ablations/_baseline.sh"

: "${SEEDS:=42 43 44 45 46}"
: "${GPU:=0}"

echo "========================================"
echo "Reproduce Table 16: HW trainability isolation cells (A/C/D)"
echo "========================================"

# --- Cell C: Protocol 1, r=1.0 (additional eval, not in main-table grid) ---
# Main-table eval covered r∈{0.1, 0.3, 0.5, 0.7}; r=1.0 needs a separate call.
# Both canonical CRAFT (C4_hw_mft_p10) and CRAFT-Core (Core_hw) are evaluated.
echo ""
echo "=== Cell C: Protocol 1 r=1.0 ==="
for prefix in "C4_hw_mft_p10" "Core_hw"; do
  for s in $SEEDS; do
    ckpt="$OUTPUT_ROOT/${prefix}_s${s}/best.ckpt"
    out_dir="$OUTPUT_ROOT/${prefix}_s${s}/eval_p1_r10"
    out_yaml="$out_dir/four_protocol_eval_results.yaml"
    [[ ! -f "$ckpt" ]] && { echo "    [skip] no ckpt $ckpt"; continue; }
    [[ -f "$out_yaml" ]] && { echo "    [skip] $out_yaml exists"; continue; }
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "    [dry] eval Cell C: ${prefix}_s${s}"
      continue
    fi
    mkdir -p "$out_dir"
    echo "    [eval gpu=$GPU] Cell C: ${prefix}_s${s}"
    CUDA_VISIBLE_DEVICES="$GPU" python incomplete_eval.py \
      --checkpoint "$ckpt" --dataset handwritten --data_dir "$DATA_DIR" \
      --protocol completer --missing_rates 1.0 \
      --num_trials 5 --seed 42 \
      --output_dir "$out_dir" \
      > "$LOG_DIR/eval_${prefix}_s${s}_p1r10.log" 2>&1
  done
done

# --- Cells A and D: already in main-table eval at r=0.7 ---
# Cell A = Protocol 1 (completer) r=0.7
# Cell D = Protocol 4 (cell_level)  r=0.7
# Both already in C4_hw_mft_p10_s*/four_protocol_eval_results.yaml; nothing to run.
echo ""
echo "=== Cells A, D: sliced from main-table eval ==="
for s in $SEEDS; do
  yaml="$OUTPUT_ROOT/C4_hw_mft_p10_s${s}/four_protocol_eval_results.yaml"
  [[ -f "$yaml" ]] && echo "    [ok] $yaml" || echo "    [MISSING] $yaml — run reproduce_main_table.sh first"
done

# --- Aggregate Cells A/C/D into one summary ---
echo ""
echo "=== Aggregate Table 16 ==="
python - <<'PY'
import yaml, glob, os, statistics
from collections import defaultdict

OUTROOT = "outputs/CRAFT_MVC"

def collect(prefix, protocol, rate, eval_subdir=None):
    accs = []
    for d in sorted(glob.glob(f"{OUTROOT}/{prefix}_s*/")):
        path = (os.path.join(d, eval_subdir, "four_protocol_eval_results.yaml")
                if eval_subdir else
                os.path.join(d, "four_protocol_eval_results.yaml"))
        if not os.path.exists(path):
            continue
        try:
            r = yaml.safe_load(open(path))
            v = r["protocols"][protocol][str(rate)]["ACC"]
            # could be a dict (mean/std) or a single float
            mean = v["mean"] if isinstance(v, dict) else v
            accs.append(float(mean))
        except (KeyError, TypeError):
            continue
    return accs

cells = [
    ("Cell A", "completer",  0.7, None),
    ("Cell C", "completer",  1.0, "eval_p1_r10"),
    ("Cell D", "cell_level", 0.7, None),
]
print(f"{'Cell':<8}  {'CRAFT-Core':<22}  {'CRAFT (canonical)':<22}")
print("-" * 60)
for name, proto, rate, sub in cells:
    core = collect("Core_hw",          proto, rate, sub)
    cano = collect("C4_hw_mft_p10",    proto, rate, sub)
    fmt = lambda xs: f"{statistics.mean(xs)*100:.2f}±{statistics.stdev(xs)*100:.2f} ({len(xs)}s)" \
                     if len(xs) >= 2 else (f"{xs[0]*100:.2f} (1s)" if xs else "—")
    print(f"{name:<8}  {fmt(core):<22}  {fmt(cano):<22}")
PY

echo "========================================"
echo "Done. Compare numbers above with paper Table 16."
echo "========================================"
