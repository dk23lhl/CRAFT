# CRAFT: Beyond Missing Rate

Anonymous code release accompanying the NeurIPS submission *"Beyond Missing Rate:
Robust Multi-View Clustering via Train-Once Transformer."*

## Repository layout

```
.
├── main_mvc.py                 # Training entry (Stage 1 + Stage 2 + optional MFT)
├── incomplete_eval.py          # 4-protocol IMVC evaluation
├── aggregate_results.py        # Aggregate training results.yaml across seeds
├── aggregate_imvc_results.py   # Aggregate four_protocol_eval_results.yaml
├── summarize_csv.py            # Pretty-print mean ± std markdown tables
├── requirements.txt
├── configs/                    # Example YAML configs
├── datasets/                   # Dataset loaders + view-mask generation
├── models/                     # View encoders + ViewFusionTransformer
├── pl_modules/                 # Lightning module (CRAFT)
├── evaluation/                 # Clustering metrics (ACC/NMI/ARI)
├── losses/
└── scripts/
    ├── reproduce_main_table.sh        # Paper Tables 2, 10, 11, 12, 13
    ├── reproduce_isolation_cell.sh    # Table 16  (Appx D.9)
    ├── reproduce_rate_matched_control.sh  # Table 14  (Appx D.6)
    ├── reproduce_fusion_spectrum.sh   # Table 19  (Appx D.11)
    ├── reproduce_component_ablation.sh# Table 20  (Appx E.1)
    ├── reproduce_craft_core.sh        # Table 21  (Appx E.2)
    ├── reproduce_sensitivity.sh       # Figure 4  (Appx E.3)
    ├── reproduce_strategy.sh          # Table 22  (Appx E.4)
    ├── reproduce_aggregation.sh       # Table 23  (Appx E.5)
    └── ablations/                     # Underlying granular training/eval scripts
```

## Environment

Single NVIDIA RTX 3090 (24 GB VRAM), CUDA 12.1. Install:

```bash
python -m venv .venv && source .venv/bin/activate   # or use conda
pip install -r requirements.txt
```

## Data

Datasets are distributed via the anonymous companion repository (link in submission).
Place dataset files under `./data/` so loaders can find them:

```
data/
├── handwritten.mat
├── cub.mat
├── multifashion.npz
├── uci-digit.mat
├── out-scene.mat
├── caltech.mat       # caltech101-20
└── youtubeface/      # or YTF.mat
```

See `datasets/load_data.py` for the exact filename each loader expects.

## Reproduction

All scripts skip ckpts that already exist (`results.yaml` / `four_protocol_eval_results.yaml`),
so re-running is idempotent. Override defaults via env vars (`SEEDS`, `GPU`, `PARALLEL`).

| Script | Paper artifact | Wall-clock (single 3090) |
|---|---|---|
| `reproduce_main_table.sh` | Tables 2, 10–13 (main + 4 appendix datasets) | ~12–18 h |
| `reproduce_rate_matched_control.sh` | Table 14 (Appx D.6) | ~3 h |
| `reproduce_isolation_cell.sh` | Table 16 (Appx D.9) — slices main-table eval | ~30 min |
| `reproduce_fusion_spectrum.sh` | Table 19 (Appx D.11) | ~6 h |
| `reproduce_component_ablation.sh` | Table 20 (Appx E.1) | ~6 h |
| `reproduce_craft_core.sh` | Table 21 (Appx E.2) | ~3 h |
| `reproduce_sensitivity.sh` | Figure 4 (Appx E.3) | ~10 h |
| `reproduce_strategy.sh` | Table 22 (Appx E.4) | ~2 h |
| `reproduce_aggregation.sh` | Table 23 (Appx E.5) | ~6 h |

Times assume `PARALLEL=4` on a single GPU; reduce if VRAM-limited.

### Quick start (smoke test, ~10 min)

```bash
SEEDS="42" GPU=0 PARALLEL=1 bash scripts/reproduce_main_table.sh
```

This trains one seed of CRAFT on each dataset (no MFT) and runs a partial eval.
Use `DRY_RUN=1` to print commands without executing.

### Full reproduction

```bash
GPU=0 PARALLEL=4 bash scripts/reproduce_main_table.sh
GPU=0 PARALLEL=4 bash scripts/reproduce_rate_matched_control.sh
GPU=0 PARALLEL=4 bash scripts/reproduce_isolation_cell.sh
GPU=0 PARALLEL=4 bash scripts/reproduce_fusion_spectrum.sh
GPU=0 PARALLEL=4 bash scripts/reproduce_component_ablation.sh
GPU=0 PARALLEL=4 bash scripts/reproduce_craft_core.sh
GPU=0 PARALLEL=4 bash scripts/reproduce_sensitivity.sh
GPU=0 PARALLEL=4 bash scripts/reproduce_strategy.sh
GPU=0 PARALLEL=4 bash scripts/reproduce_aggregation.sh
```

Each produces a `*_summary.md` at repo root with mean ± std numbers to compare
against the corresponding paper table.

### Wall-clock measurement (Appx D.8)

Per-method training times are logged at `logs/ablation/<exp_name>.log`. Aggregate via:

```bash
for f in logs/ablation/C8_mf_mfth1024_xlong_s*.log; do
  grep -E "elapsed|Training time" "$f" | tail -1
done
```

The 8.8× advantage on MultiFashion compares (16 baseline retraining runs × 8 min)
to a single 14.5-min CRAFT run.

## Single training command

For a one-off CRAFT run on any of the 7 datasets:

```bash
python main_mvc.py \
  --dataset handwritten \                # or cub / multifashion / caltech101-20 / uci-digit / out-scene / youtubeface
  --data_dir ./data \
  --encoder_type deep --encoder_hidden_dim 512 --embed_dim 256 \
  --cluster_temperature 0.1 --lambda_repr 0.5 --lambda_entropy 5.0 \
  --stage1_epochs 100 --max_epochs 200 --stage2_lr 1e-5 \
  --seed 42 --exp_name my_run
```

Per-dataset hyperparameters live in `scripts/ablations/_baseline.sh` (functions
`hw_args`, `cub_args`, `mfashion_args`, `caltech_args`, `uci_args`, `outscene_args`,
`ytf_args`); see that file for the canonical configuration of each dataset.

## Single evaluation command

```bash
python incomplete_eval.py \
  --checkpoint outputs/CRAFT_MVC/<exp_name>/best.ckpt \
  --dataset handwritten --data_dir ./data \
  --protocol all --missing_rates 0.1 0.3 0.5 0.7 \
  --num_trials 5 --seed 42
```

This produces `four_protocol_eval_results.yaml` next to the checkpoint with ACC/NMI/ARI
under all four masking protocols (Tables 1–14 of the paper).

## Baselines

Baseline numbers in Table 2 are obtained from each method's official repository
under the configurations described in Appendix A.3. Their training code is not
redistributed here.

## License

To be released upon acceptance.
