"""
Consume CSVs from aggregate_results.py / aggregate_imvc_results.py and produce
mean+/-std markdown tables grouped by (dataset, variant) or
(dataset, variant, protocol, missing_rate) over seeds.

variant is the exp_name with the trailing `_s<seed>` removed, e.g.:
    B8_hw_concat_mlp_s43       -> variant = B8_hw_concat_mlp
    A2_cub_attention_mask_s44  -> variant = A2_cub_attention_mask

Schema is auto-detected: CSV with a missing_rate column -> IMVC mode; else training mode.
Default metrics:
    training CSV -> final_acc / final_nmi / final_ari
    IMVC CSV     -> acc_mean / nmi_mean / ari_mean

Usage:
    python summarize_csv.py --csv b8.csv
    python summarize_csv.py --csv b1.csv b3.csv b6.csv --out summary.md
    python summarize_csv.py --csv imvc_b8_a2.csv --decimals 4
    python summarize_csv.py --csv b4.csv --metrics final_acc      # ACC only
"""

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


def strip_seed(name: str) -> str:
    return re.sub(r'_s\d+$', '', name or '')


def parse_float(x):
    if x is None or x == '':
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fmt_mean_std(values, decimals=4):
    vals = [v for v in values if v is not None]
    if not vals:
        return ''
    if len(vals) == 1:
        return f"{vals[0]:.{decimals}f}"
    m = statistics.mean(vals)
    s = statistics.pstdev(vals) if len(vals) < 2 else statistics.stdev(vals)
    return f"{m:.{decimals}f}±{s:.{decimals}f}"


def detect_schema(rows):
    cols = set(rows[0].keys())
    is_imvc = 'missing_rate' in cols and 'acc_mean' in cols
    return is_imvc


def default_metrics(is_imvc):
    return ['acc_mean', 'nmi_mean', 'ari_mean'] if is_imvc else ['final_acc', 'final_nmi', 'final_ari']


def sort_key(row, is_imvc):
    # row: [variant, (protocol, mr), metrics..., n]
    variant = row[0]
    if not is_imvc:
        return (variant,)
    proto = row[1]
    mr_str = row[2]
    try:
        mr_num = float(mr_str)
    except (TypeError, ValueError):
        mr_num = float('inf')
    return (variant, proto, mr_num)


def summarize_one(csv_path: Path, metrics_override=None, decimals=4):
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        return [f"## `{csv_path.name}` — empty\n"]

    is_imvc = detect_schema(rows)
    metrics = metrics_override or default_metrics(is_imvc)

    # key: (dataset, variant, protocol, mr) → metric → [values]
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        ds = r.get('dataset') or '?'
        variant = strip_seed(r.get('exp_name', ''))
        proto = r.get('protocol', '') if is_imvc else ''
        mr = r.get('missing_rate', '') if is_imvc else ''
        key = (ds, variant, proto, mr)
        for m in metrics:
            groups[key][m].append(parse_float(r.get(m)))

    by_dataset = defaultdict(list)
    for (ds, variant, proto, mr), m_dict in groups.items():
        n = max((len([v for v in vs if v is not None]) for vs in m_dict.values()), default=0)
        cells = [fmt_mean_std(m_dict[m], decimals) for m in metrics]
        if is_imvc:
            by_dataset[ds].append([variant, proto, mr] + cells + [str(n)])
        else:
            by_dataset[ds].append([variant] + cells + [str(n)])

    if is_imvc:
        header = ['variant', 'protocol', 'missing_rate'] + metrics + ['n']
    else:
        header = ['variant'] + metrics + ['n']

    lines = [f"## `{csv_path.name}` ({'IMVC' if is_imvc else 'training'})", ""]
    for ds in sorted(by_dataset):
        lines.append(f"### dataset: `{ds}`")
        lines.append("")
        lines.append('| ' + ' | '.join(header) + ' |')
        lines.append('| ' + ' | '.join(['---'] * len(header)) + ' |')
        rows_ds = sorted(by_dataset[ds], key=lambda r: sort_key(r, is_imvc))
        for row in rows_ds:
            lines.append('| ' + ' | '.join(row) + ' |')
        lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', nargs='+', required=True, help='One or more CSV paths')
    ap.add_argument('--out', default=None, help='Output markdown path; prints to stdout if not set')
    ap.add_argument('--metrics', nargs='+', default=None,
                    help='Override default metric columns; training CSV defaults to '
                         'final_acc/nmi/ari, IMVC defaults to acc_mean/nmi_mean/ari_mean')
    ap.add_argument('--decimals', type=int, default=4)
    args = ap.parse_args()

    all_lines = ['# Ablation summary', '']
    for path in args.csv:
        p = Path(path)
        if not p.exists():
            print(f"[warn] not found: {p}")
            continue
        all_lines.extend(summarize_one(p, args.metrics, args.decimals))

    out = '\n'.join(all_lines)
    if args.out:
        Path(args.out).write_text(out, encoding='utf-8')
        print(f"wrote {args.out}")
    else:
        print(out)


if __name__ == '__main__':
    main()
