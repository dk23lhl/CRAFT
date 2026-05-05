# B9 SetMLP + B8-MF launch plan (2 GPU x 5 parallel)

## Workload

| Job | Trainings | Dataset | epochs | Note |
|---|---|---|---|---|
| B9 SetMLP x HW x 5 seeds | 5 | HW (V=6) | 200 | h=192 |
| B9 SetMLP x CUB x 5 seeds | 5 | CUB (V=2) | 200 | h=128 |
| B9 SetMLP x MF x 5 seeds | 5 | MF (V=3) | 400 | h=144 |
| B8 ConcatMLP x MF x 5 seeds | 5 | MF (V=3) | 400 | fills B8 spectrum |
| **Total** | **20 trains** | | | |

Eval: 20 ckpts x 4 protocols x 4 rates x 5 trials (IMVC eval per ckpt is tens of seconds to a few minutes).

## Pre-launch sanity check (local or server, ~10 seconds)

```bash
python scripts/sanity_setmlp.py
```

Only enter training if all four checks pass. Any failure -> stop and investigate.

## Two-GPU split (balanced: HW+CUB on GPU 0; MF SetMLP+ConcatMLP on GPU 1)

Each GPU runs 10 trainings x 5 parallel = 2 waves. HW/CUB are 200 epochs, MF is 400 epochs, so GPU 1's
wall-clock is roughly 2x GPU 0's. To balance better, move MF ConcatMLP to GPU 0 (see alt scheme).

### Default split

**Terminal 1 (GPU 0, HW + CUB SetMLP) -- ~2x HW training time**
```bash
GPU=0 PARALLEL=5 DATASETS="hw cub" bash scripts/ablations/run_b9_setmlp.sh
```

**Terminal 2 (GPU 1, MF SetMLP + MF ConcatMLP) -- ~2x MF training time (about 4x HW)**
```bash
GPU=1 PARALLEL=5 DATASETS="mf" bash scripts/ablations/run_b9_setmlp.sh && \
GPU=1 PARALLEL=5 bash scripts/ablations/run_b8_mf.sh
```

### Alt split (more balanced: HW+ConcatMLP-MF on one side, CUB+SetMLP-MF on the other)

```bash
# Terminal 1 (GPU 0)
GPU=0 PARALLEL=5 DATASETS="hw" bash scripts/ablations/run_b9_setmlp.sh && \
GPU=0 PARALLEL=5 bash scripts/ablations/run_b8_mf.sh

# Terminal 2 (GPU 1)
GPU=1 PARALLEL=5 DATASETS="cub mf" bash scripts/ablations/run_b9_setmlp.sh
```

GPU 0 wall: HW (200e) + MF Concat (400e) = ~600 epoch units
GPU 1 wall: CUB (200e) + MF SetMLP (400e) = ~600 epoch units
Both sides have equal length. **Recommended: use the alt split**.

## Eval phase (after all training finishes)

```bash
# Terminal 1 (GPU 0): HW + CUB
GPU=0 PARALLEL=5 PREFIXES="B9_hw_ B9_cub_" bash scripts/ablations/run_eval_b9_setmlp.sh

# Terminal 2 (GPU 1): MF (SetMLP + ConcatMLP)
GPU=1 PARALLEL=5 PREFIXES="B9_mf_ B8_mf_" bash scripts/ablations/run_eval_b9_setmlp.sh
```

## Aggregate results

```bash
python aggregate_results.py --filter B9_ --out b9.csv
python aggregate_results.py --filter 'B8_mf_' --out b8_mf.csv
python aggregate_imvc_results.py --filter B9_,B8_mf_ --out imvc_b9_b8mf.csv
python summarize_csv.py --csv imvc_b9_b8mf.csv --out b9_b8mf_imvc_summary.md
```

## Important caveat: capacity-matched hidden values

Default hidden values in the script (`SETMLP_HIDDEN_HW=192`, `_CUB=128`, `_MF=144`) are computed
assuming `d=embed_dim=128` ConcatMLP capacity. But `_baseline.sh` actually uses:
- `hw_args` -> `embed_dim=256`
- `cub_args` -> `embed_dim=128`
- `mfashion_args` -> `embed_dim=256`

To strictly match `_baseline.sh`'s ConcatMLP (i.e. d=256/128/256, h_default=2d), use:
```bash
SETMLP_HIDDEN_HW=560 SETMLP_HIDDEN_CUB=168 SETMLP_HIDDEN_MF=400 \
  GPU=0 PARALLEL=5 DATASETS="hw cub mf" bash scripts/ablations/run_b9_setmlp.sh
```

> Note: hidden values below capacity-match are equivalent to "SetMLP running at smaller capacity",
> which makes the result more likely to fall into Section 6.3 outcome (c). This is a conservative
> baseline setting; if instead you want to see "whether SetMLP can still match at the capacity ceiling",
> use the alt values.

## Smoke (40 min) -- optional early kill switch

To validate that MF V=3 r=0.7 doesn't collapse (the highest risk point), run a single trial:
```bash
SEEDS="42" DATASETS="mf" PARALLEL=1 GPU=0 \
  bash scripts/ablations/run_b9_setmlp.sh
```
Once finished, evaluate this single ckpt immediately:
```bash
CUDA_VISIBLE_DEVICES=0 python incomplete_eval.py \
  --checkpoint outputs/CRAFT_MVC/B9_mf_setmlp_h144_s42/best.ckpt \
  --dataset multifashion --data_dir ./data \
  --protocol per_view_ind --missing_rates 0.7 \
  --num_trials 5 --seed 42
```
If ACC > 0.5 (no collapse), continue with the full suite; if < 0.3, stop and fall back to
cardinality-embedding.
