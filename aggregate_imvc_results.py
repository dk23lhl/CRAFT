"""
Aggregate IMVC (cell-level) evaluation results: walk
./outputs/CRAFT_MVC/*/four_protocol_eval_results.yaml, flatten by
(exp, missing_rate, protocol), join with training config, and export CSV.

Usage:
    python aggregate_imvc_results.py
    python aggregate_imvc_results.py --filter B8_,A2_ --out imvc.csv
    python aggregate_imvc_results.py --protocol cell_level --out imvc_cell.csv

One row per (exp, protocol, missing_rate). complete_acc/nmi/ari are the same
checkpoint's complete-view metrics (from the 'complete' field of the same
four_protocol_eval_results.yaml) so ACC drop can be read off easily.
"""

import argparse
import csv
from pathlib import Path

import yaml


CONFIG_KEYS = [
    'dataset', 'seed',
    'fusion_type', 'fusion_n_layers',
    'embed_dim', 'cluster_temperature',
    'lambda_cluster', 'lambda_entropy', 'lambda_repr', 'lambda_recon',
    'mask_strategy',
    'stage1_epochs', 'max_epochs',
    'train_mask_prob',
]

METRIC_COLUMNS = [
    'protocol', 'missing_rate',
    'real_cell_mr', 'complete_sample_ratio', 'avg_views_per_sample',
    'acc_mean', 'acc_std', 'nmi_mean', 'nmi_std', 'ari_mean', 'ari_std',
    'complete_acc', 'complete_nmi', 'complete_ari',
    'acc_drop', 'acc_drop_pct',
]


def safe_load_yaml(path: Path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"  [warn] failed to load {path}: {e}")
        return None


def extract_rows(run_dir: Path, protocol_filter=None, eval_subdir=None):
    cfg_path = run_dir / 'config.yaml'
    if eval_subdir:
        eval_path = run_dir / eval_subdir / 'four_protocol_eval_results.yaml'
        # Insert __<subdir> before _s<seed> so summarize_csv.py:strip_seed still trims correctly,
        # e.g. C3_best_hw_s42 + eval_attention_mask -> C3_best_hw__eval_attention_mask_s42
        import re as _re
        m = _re.search(r'_s\d+$', run_dir.name)
        if m:
            exp_label = f"{run_dir.name[:m.start()]}__{eval_subdir}{m.group()}"
        else:
            exp_label = f"{run_dir.name}__{eval_subdir}"
    else:
        eval_path = run_dir / 'four_protocol_eval_results.yaml'
        exp_label = run_dir.name
    if not eval_path.exists():
        return []

    cfg = safe_load_yaml(cfg_path) if cfg_path.exists() else {}
    ev = safe_load_yaml(eval_path)
    if ev is None:
        return []

    base = {'exp_name': exp_label, 'run_dir': str(run_dir)}
    for k in CONFIG_KEYS:
        base[k] = (cfg or {}).get(k)
    # eval-time mask_strategy override (incomplete_eval.py writes it at eval yaml top level)
    eval_strat = ev.get('mask_strategy')
    if eval_strat is not None:
        base['mask_strategy'] = eval_strat

    complete = ev.get('complete', {}) or {}
    c_acc = complete.get('ACC')
    c_nmi = complete.get('NMI')
    c_ari = complete.get('ARI')

    rows = []
    protocols = ev.get('protocols', {}) or {}
    for proto, by_rate in protocols.items():
        if protocol_filter and proto != protocol_filter:
            continue
        if not isinstance(by_rate, dict):
            continue
        for rate_str, r in by_rate.items():
            if not isinstance(r, dict):
                continue
            try:
                rate = float(rate_str)
            except (TypeError, ValueError):
                rate = rate_str
            acc = r.get('ACC', {}) or {}
            nmi = r.get('NMI', {}) or {}
            ari = r.get('ARI', {}) or {}
            acc_mean = acc.get('mean')
            drop = (c_acc - acc_mean) if (c_acc is not None and acc_mean is not None) else None
            drop_pct = (drop / c_acc * 100) if (drop is not None and c_acc) else None

            row = dict(base)
            row.update({
                'protocol': proto,
                'missing_rate': rate,
                'real_cell_mr': r.get('cell_missing_rate'),
                'complete_sample_ratio': r.get('complete_sample_ratio'),
                'avg_views_per_sample': r.get('avg_views_per_sample'),
                'acc_mean': acc_mean,
                'acc_std': acc.get('std'),
                'nmi_mean': nmi.get('mean'),
                'nmi_std': nmi.get('std'),
                'ari_mean': ari.get('mean'),
                'ari_std': ari.get('std'),
                'complete_acc': c_acc,
                'complete_nmi': c_nmi,
                'complete_ari': c_ari,
                'acc_drop': drop,
                'acc_drop_pct': drop_pct,
            })
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default='./outputs/CRAFT_MVC')
    ap.add_argument('--out', type=str, default='imvc_summary.csv')
    ap.add_argument('--filter', type=str, default=None,
                    help='exp_name prefix(es), comma-separated (e.g. B8_,A2_)')
    ap.add_argument('--protocol', type=str, default=None,
                    help='Only export the given protocol (e.g. cell_level); default: all')
    ap.add_argument('--eval_subdir', type=str, default=None,
                    help='Read <run_dir>/<eval_subdir>/four_protocol_eval_results.yaml, '
                         'used for aggregating strategy-override evaluations '
                         '(e.g. eval_attention_mask). Default reads from run_dir top level.')
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    prefixes = None
    if args.filter:
        prefixes = tuple(p.strip() for p in args.filter.split(',') if p.strip())

    run_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if prefixes:
        run_dirs = [d for d in run_dirs if d.name.startswith(prefixes)]

    print(f"scanning {len(run_dirs)} run dirs under {root}"
          f"{f' (filter prefixes={prefixes})' if prefixes else ''}"
          f"{f' (protocol={args.protocol})' if args.protocol else ''}")

    all_rows = []
    n_no_eval = 0
    for d in run_dirs:
        rows = extract_rows(d, protocol_filter=args.protocol, eval_subdir=args.eval_subdir)
        if not rows:
            n_no_eval += 1
            continue
        all_rows.extend(rows)

    if not all_rows:
        print("no IMVC results found.")
        return

    columns = (['exp_name'] + CONFIG_KEYS + METRIC_COLUMNS + ['run_dir'])
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k) for k in columns})

    print(f"wrote {len(all_rows)} rows to {args.out} "
          f"(from {len(run_dirs) - n_no_eval} ckpts, "
          f"{n_no_eval} dirs missing four_protocol_eval_results.yaml)")

    by_proto = {}
    for row in all_rows:
        p = row.get('protocol') or '?'
        by_proto[p] = by_proto.get(p, 0) + 1
    print("by protocol:")
    for p, n in sorted(by_proto.items()):
        print(f"  {p}: {n}")


if __name__ == '__main__':
    main()
