"""
Aggregate ablation results: walk ./outputs/CRAFT_MVC/*/ for config.yaml + results.yaml
and export a single CSV for pivoting by dataset/variant/seed.

Usage:
    python aggregate_results.py
    python aggregate_results.py --root ./outputs/CRAFT_MVC --out results_summary.csv
    python aggregate_results.py --filter B8_          # only runs whose exp_name starts with B8_

Output columns:
    exp_name, dataset, seed, fusion_type, fusion_n_layers, embed_dim,
    cluster_temperature, lambda_*, mask_strategy, stage1_epochs, max_epochs,
    skip_stage2, missing_rate, train_mask_prob, selected_views,
    final_acc, final_nmi, final_ari, best_epoch, final_source, run_dir
"""

import argparse
import csv
import os
from pathlib import Path

import yaml


CONFIG_KEYS = [
    'dataset', 'seed',
    'fusion_type', 'fusion_n_layers', 'fusion_n_heads',
    'embed_dim', 'encoder_hidden_dim', 'encoder_type',
    'cluster_temperature',
    'lambda_cluster', 'lambda_entropy', 'lambda_repr', 'lambda_recon',
    'mask_strategy',
    'stage1_epochs', 'max_epochs', 'stage2_lr', 'stage2_freeze',
    'skip_stage2',
    'missing_rate', 'train_mask_prob', 'num_masked_views',
    'selected_views',
    'noise_std', 'dropout_rate', 'lr',
]

METRIC_COLUMNS = ['final_acc', 'final_nmi', 'final_ari', 'best_epoch', 'final_source']


def safe_load_yaml(path: Path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"  [warn] failed to load {path}: {e}")
        return None


def extract_row(run_dir: Path):
    cfg_path = run_dir / 'config.yaml'
    res_path = run_dir / 'results.yaml'
    if not cfg_path.exists() or not res_path.exists():
        return None

    cfg = safe_load_yaml(cfg_path)
    res = safe_load_yaml(res_path)
    if cfg is None or res is None:
        return None

    row = {'exp_name': run_dir.name, 'run_dir': str(run_dir)}
    for k in CONFIG_KEYS:
        row[k] = cfg.get(k)

    final = res.get('final', {}) or {}
    row['final_acc'] = final.get('ACC')
    row['final_nmi'] = final.get('NMI')
    row['final_ari'] = final.get('ARI')
    row['best_epoch'] = res.get('best_epoch')
    row['final_source'] = res.get('final_source')
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default='./outputs/CRAFT_MVC',
                    help='Experiment output root directory')
    ap.add_argument('--out', type=str, default='results_summary.csv',
                    help='Output CSV path')
    ap.add_argument('--filter', type=str, default=None,
                    help='Only aggregate runs whose exp_name starts with this prefix (e.g. B8_)')
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    run_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if args.filter:
        run_dirs = [d for d in run_dirs if d.name.startswith(args.filter)]

    print(f"scanning {len(run_dirs)} run dirs under {root}"
          f"{f' (filter prefix={args.filter})' if args.filter else ''}")

    rows = []
    skipped = 0
    for d in run_dirs:
        row = extract_row(d)
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    if not rows:
        print("no results found.")
        return

    columns = ['exp_name'] + CONFIG_KEYS + METRIC_COLUMNS + ['run_dir']
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in columns})

    print(f"wrote {len(rows)} rows to {args.out} (skipped {skipped} incomplete dirs)")

    by_dataset = {}
    for row in rows:
        ds = row.get('dataset') or '?'
        by_dataset[ds] = by_dataset.get(ds, 0) + 1
    print("by dataset:")
    for ds, n in sorted(by_dataset.items()):
        print(f"  {ds}: {n}")


if __name__ == '__main__':
    main()