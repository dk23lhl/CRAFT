"""
Incomplete Multi-View Evaluation for CRAFT — Four Protocol Version (v2)

Changes vs v1:
- After loading ckpt, immediately print stats for mask_token / cls_token / view_embed
  to diagnose whether mask_token was ever trained in a complete-trained model
- The summary table explicitly prints true cell MR and avg views per sample
- An extra "real cell MR view" table is emitted after the summary for easy alignment with Energy-DIMC

Supports comparing four masking protocols, used by the IMVC evaluation audit paper.

Protocols:
  completer      — true COMPLETER: pick m samples, drop 1 view from each
  dcg_original   — DCG/ProImp/APADC actual implementation: mr/V per-cell
  per_view_ind   — Per-view independent: each cell drops independently with prob mr
  cell_level     — standardized cell-level: precisely controls cell missing rate = mr

Usage:
    # Full protocol comparison (4 x 4 = 16 combinations)
    python incomplete_eval.py \
        --checkpoint ./outputs/CRAFT_MVC/best_model.ckpt \
        --dataset handwritten \
        --data_dir ./data \
        --missing_rates 0.1 0.3 0.5 0.7 \
        --protocol all \
        --num_trials 5 --seed 42

    # Single-protocol single-mr quick sanity check
    python incomplete_eval.py \
        --checkpoint ./outputs/CRAFT_MVC/best_model.ckpt \
        --dataset handwritten \
        --data_dir ./data \
        --missing_rates 0.3 \
        --protocol dcg_original \
        --num_trials 1 --seed 42

    # V=2 subset
    python incomplete_eval.py \
        --checkpoint ./outputs/CRAFT_MVC/hw_2v.ckpt \
        --dataset handwritten \
        --data_dir ./data \
        --selected_views 1,4 \
        --protocol all --num_trials 3
"""

import os
import argparse
import numpy as np
import torch
import yaml
from collections import defaultdict

from datasets.load_data import load_dataset
from datasets.multiview_dataset import MultiViewDataModule
from models.view_encoder import build_view_encoders, build_view_decoders
from models.fusion import ViewFusionTransformer, MultiViewEncoder
from pl_modules.craft_mvc import CRAFT
from evaluation.clustering_metrics import evaluate_clustering


def generate_incomplete_mask(num_samples: int,
                             num_views: int,
                             missing_rate: float,
                             protocol: str = "completer",
                             rng: np.random.Generator = None
                             ) -> np.ndarray:
    """Generate a view missing mask under one of four protocols.

    Returns:
        mask: [N, V] bool ndarray, True = observed, False = missing
    """
    if rng is None:
        rng = np.random.default_rng()

    if missing_rate == 0.0:
        return np.ones((num_samples, num_views), dtype=bool)

    if protocol == "completer":
        return _mask_completer(num_samples, num_views, missing_rate, rng)
    elif protocol == "dcg_original":
        return _mask_dcg_original(num_samples, num_views, missing_rate, rng)
    elif protocol == "per_view_ind":
        return _mask_per_view_independent(num_samples, num_views, missing_rate, rng)
    elif protocol == "cell_level":
        return _mask_cell_level(num_samples, num_views, missing_rate, rng)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")


def _mask_completer(n, v, mr, rng):
    """True COMPLETER: pick m=mr*n samples, drop exactly 1 view from each.
    Actual cell missing rate = mr / V.
    """
    mask = np.ones((n, v), dtype=bool)
    m = int(round(n * mr))
    if m == 0:
        return mask
    selected = rng.choice(n, size=m, replace=False)
    deleted_views = rng.integers(0, v, size=m)
    mask[selected, deleted_views] = False
    return mask


def _mask_dcg_original(n, v, mr, rng):
    """DCG/ProImp/APADC actual code: mr/V per-cell independent.
    Actual cell missing rate ~ mr / V (same number as COMPLETER but different mechanism).
    Guarantees at least 1 view kept per sample via the fallback mechanism.
    """
    effective_mr = mr / v
    one_rate = 1.0 - effective_mr

    if one_rate <= (1 / v):
        view_preserve = np.zeros((n, v), dtype=bool)
        chosen = rng.integers(0, v, size=n)
        view_preserve[np.arange(n), chosen] = True
        return view_preserve

    if one_rate == 1:
        return np.ones((n, v), dtype=bool)

    # DCG's iterative calibration loop
    error = 1.0
    while error >= 0.005:
        view_preserve = np.zeros((n, v), dtype=bool)
        chosen = rng.integers(0, v, size=n)
        view_preserve[np.arange(n), chosen] = True

        one_num = v * n * one_rate - n
        ratio = one_num / (v * n)
        matrix_iter = (rng.integers(0, 100, size=(n, v)) < int(ratio * 100))

        a = np.sum((matrix_iter & view_preserve).astype(int))
        if a == one_num:
            one_num_iter = one_num
        else:
            one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (v * n)
        matrix_iter = (rng.integers(0, 100, size=(n, v)) < int(ratio * 100))

        matrix = (matrix_iter | view_preserve)
        ratio = np.sum(matrix) / (v * n)
        error = abs(one_rate - ratio)

    return matrix


def _mask_per_view_independent(n, v, mr, rng):
    """Per-view independent: each (sample, view) cell drops independently with prob mr.
    Actual cell missing rate ~ mr.
    Guarantees at least 1 view per sample.
    """
    mask = rng.random((n, v)) >= mr
    all_missing = np.where(mask.sum(axis=1) == 0)[0]
    for i in all_missing:
        mask[i, rng.integers(0, v)] = True
    return mask


def _mask_cell_level(n, v, mr, rng):
    """Standardized cell-level: precisely controls cell missing rate = mr.
    Always keeps at least 1 view per sample, so V=2 caps mr at 50%.
    """
    total_cells = n * v
    n_missing = int(round(total_cells * mr))
    max_missing = n * (v - 1)
    n_missing = min(n_missing, max_missing)

    protected = rng.integers(0, v, size=n)
    candidates = []
    for i in range(n):
        for vi in range(v):
            if vi != protected[i]:
                candidates.append((i, vi))

    rng.shuffle(candidates)
    n_missing = min(n_missing, len(candidates))

    mask = np.ones((n, v), dtype=bool)
    for idx in range(n_missing):
        i, vi = candidates[idx]
        mask[i, vi] = False

    return mask


def compute_mask_stats(mask):
    """Compute mask statistics."""
    n, v = mask.shape
    total = n * v
    observed = mask.sum()
    missing = total - observed
    cell_mr = missing / total
    per_sample_views = mask.sum(axis=1)

    # Distribution of "how many views missing" per sample
    missing_per_sample = v - per_sample_views
    missing_dist = {}
    for k in range(v + 1):
        missing_dist[k] = int((missing_per_sample == k).sum())

    complete_samples = int((missing_per_sample == 0).sum())

    return {
        "n_samples": n,
        "n_views": v,
        "total_cells": total,
        "observed_cells": int(observed),
        "missing_cells": int(missing),
        "cell_missing_rate": float(cell_mr),
        "complete_sample_ratio": float(complete_samples / n),
        "samples_with_any_missing": int((per_sample_views < v).sum()),
        "min_views_per_sample": int(per_sample_views.min()),
        "avg_views_per_sample": float(per_sample_views.mean()),
        "missing_distribution": missing_dist,
    }


def diagnose_learnable_tokens(model):
    """Print stats for mask_token / cls_token / view_embed.

    Heuristic:
    - If norm ~ 0.02 * sqrt(embed_dim) (e.g. ~0.32 for embed_dim=256), the parameter
      is still near init and very likely never got a forward-pass gradient.
    - If norm is much larger (e.g. >1), the parameter has been trained.
    """
    print("\n" + "="*70)
    print("Learnable Token Diagnostics")
    print("="*70)
    print("Expected init norm (std=0.02): ")
    print("  embed_dim=128 -> ~0.23")
    print("  embed_dim=256 -> ~0.32")
    print("  embed_dim=512 -> ~0.45")
    print("If mask_token norm is close to init -> never trained in forward pass")
    print("-"*70)

    tokens_to_check = ['mask_token', 'cls_token', 'view_embed']
    found_any = False

    for name, param in model.named_parameters():
        if any(t in name for t in tokens_to_check):
            found_any = True
            p = param.detach().cpu().float()
            n_el = p.numel()
            print(f"  {name}")
            print(f"    shape   : {list(p.shape)}")
            print(f"    mean    : {p.mean().item():+.6f}")
            print(f"    std     : {p.std().item():.6f}")
            print(f"    min/max : {p.min().item():+.6f} / {p.max().item():+.6f}")
            print(f"    L2 norm : {p.norm().item():.6f}")
            print(f"    L2/sqrt(numel): {(p.norm().item() / (n_el ** 0.5)):.6f}")
            print(f"    requires_grad : {param.requires_grad}")
            print()

    if not found_any:
        print("  [Warning] No mask_token / cls_token / view_embed found")
        print("  This may indicate a different model architecture")

    print("="*70 + "\n")


@torch.no_grad()
def evaluate_incomplete(model, data_loader, device, view_mask_full):
    """Evaluate clustering performance under view-missing conditions."""
    model.eval()
    all_preds = []
    all_labels = []
    sample_idx = 0

    for batch in data_loader:
        if len(batch) == 2:
            views, labels = batch
        elif len(batch) == 3:
            # full_dataset with augment=False may return (views, labels, vm). Our full_dataset
            # defaults to missing_rate=0 so vm=None and we hit the len==2 branch; keep this for compat.
            views, labels, _ = batch
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        B = labels.shape[0]
        views = [v.to(device) for v in views]

        batch_mask = torch.tensor(
            view_mask_full[sample_idx: sample_idx + B],
            dtype=torch.bool, device=device
        )
        sample_idx += B

        encoded = model.encoder.encode_views(views)
        z = model.encoder.fuse(encoded, view_mask=batch_mask)

        p = model.cluster_head(z)
        preds = torch.argmax(p, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    return evaluate_clustering(np.array(all_labels), np.array(all_preds))


def load_model_from_checkpoint(ckpt_path, dataset, data_dir,
                               embed_dim=None, encoder_hidden_dim=None,
                               selected_views=None, mask_strategy=None):
    """Load the full model from a checkpoint."""
    import yaml as _yaml
    from main_mvc import build_model

    features_list, labels = load_dataset(dataset, data_dir)

    if selected_views:
        view_indices = [int(x) for x in selected_views.split(',')]
        features_list = [features_list[i] for i in view_indices]
        print(f"  Selected views: {view_indices}, dims: {[f.shape[1] for f in features_list]}")

    num_views = len(features_list)
    view_dims = [f.shape[1] for f in features_list]
    num_clusters = len(np.unique(labels))

    ckpt_dir = os.path.dirname(ckpt_path)
    search_dir = ckpt_dir
    cfg = None
    for _ in range(5):
        config_path = os.path.join(search_dir, 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = _yaml.safe_load(f)
            print(f"[Info] Loaded config from: {config_path}")
            break
        search_dir = os.path.dirname(search_dir)

    if cfg is None:
        print("[Warning] config.yaml not found, using CLI args / defaults")
        cfg = {
            'embed_dim': embed_dim or 128,
            'encoder_hidden_dim': encoder_hidden_dim or 256,
        }

    if embed_dim is not None:
        cfg['embed_dim'] = embed_dim
    if encoder_hidden_dim is not None:
        cfg['encoder_hidden_dim'] = encoder_hidden_dim
    if mask_strategy is not None:
        cfg['mask_strategy'] = mask_strategy
        print(f"[Info] mask_strategy overridden to: {mask_strategy}")

    model = build_model(num_views, view_dims, num_clusters, cfg)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']

    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith('masker.') and 'asymmetric_consistency_loss' not in k
    }

    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    if missing:
        print(f"[Warning] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[Info] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    return model, features_list, labels, num_views


def run_single_protocol(model, full_loader, device, num_samples, num_views,
                        protocol, missing_rates, num_trials, seed,
                        complete_metrics):
    """Run all missing rates for a single protocol."""
    results = {}

    for rate in missing_rates:
        if rate == 0.0:
            continue

        print(f"\n  --- mr={rate}, protocol={protocol} ---")
        trial_results = defaultdict(list)
        rng_base = np.random.default_rng(seed)

        last_stats = None
        for trial in range(num_trials):
            trial_seed = rng_base.integers(0, 2**31)
            rng = np.random.default_rng(trial_seed)

            mask = generate_incomplete_mask(
                num_samples=num_samples,
                num_views=num_views,
                missing_rate=rate,
                protocol=protocol,
                rng=rng
            )

            stats = compute_mask_stats(mask)
            last_stats = stats
            metrics = evaluate_incomplete(model, full_loader, device, mask)

            for k, v in metrics.items():
                trial_results[k].append(v)

            if trial == 0:
                print(f"    [Mask] cell_mr={stats['cell_missing_rate']:.4f}, "
                      f"missing={stats['missing_cells']}/{stats['total_cells']}, "
                      f"avg_views={stats['avg_views_per_sample']:.2f}/{num_views}, "
                      f"complete_ratio={stats['complete_sample_ratio']:.4f}")
                print(f"    [Mask] missing_dist: {stats['missing_distribution']}")

            print(f"    Trial {trial+1}: ACC={metrics['ACC']:.4f} NMI={metrics['NMI']:.4f} ARI={metrics['ARI']:.4f}")

        summary = {}
        for k in ['ACC', 'NMI', 'ARI']:
            vals = trial_results[k]
            summary[k] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
            }

        acc_drop = complete_metrics['ACC'] - summary['ACC']['mean']
        acc_drop_pct = acc_drop / complete_metrics['ACC'] * 100
        print(f"    >>> ACC={summary['ACC']['mean']:.4f}+/-{summary['ACC']['std']:.4f}  "
              f"(drop: -{acc_drop_pct:.1f}%)")

        results[str(rate)] = summary
        # Save the last trial's stats (mask differs per trial but the stats are stable at this scale)
        results[str(rate)]['cell_missing_rate'] = last_stats['cell_missing_rate']
        results[str(rate)]['complete_sample_ratio'] = last_stats['complete_sample_ratio']
        results[str(rate)]['avg_views_per_sample'] = last_stats['avg_views_per_sample']

    return results


def main():
    parser = argparse.ArgumentParser(description='IMVC Four-Protocol Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='handwritten',
                        choices=['caltech101-20', 'handwritten', 'ngs', 'bbc',
                                 'uci-digit', 'hdigit', 'aloi-100', 'out-scene',
                                 'scene-15', 'landuse-21', 'cub',
                                 'multifashion', 'noisymnist-30k'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--missing_rates', type=float, nargs='+',
                        default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument('--protocol', type=str, default='all',
                        choices=['completer', 'dcg_original', 'per_view_ind',
                                 'cell_level', 'all'])
    parser.add_argument('--num_trials', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--embed_dim', type=int, default=None)
    parser.add_argument('--encoder_hidden_dim', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--repr_mode', type=str, default='simsiam',
                        choices=['simsiam', 'vicreg'])
    parser.add_argument('--selected_views', type=str, default=None,
                        help='Comma-separated view indices, e.g. "1,4" means use only Fou+KAR (default uses all views)')
    parser.add_argument('--mask_strategy', type=str, default=None,
                        choices=['learnable', 'zero', 'mean', 'attention_mask'],
                        help='Override mask_strategy from config.yaml (for ablation experiments)')

    args = parser.parse_args()
    device = torch.device('cuda' if args.gpus > 0 and torch.cuda.is_available() else 'cpu')

    print(f"\nLoading model from: {args.checkpoint}")
    model, features_list, labels, num_views = load_model_from_checkpoint(
        args.checkpoint, args.dataset, args.data_dir,
        embed_dim=args.embed_dim, encoder_hidden_dim=args.encoder_hidden_dim,
        selected_views=args.selected_views, mask_strategy=args.mask_strategy
    )
    model = model.to(device)
    model.eval()

    diagnose_learnable_tokens(model)

    num_samples = len(labels)
    num_clusters = len(np.unique(labels))
    view_dims = [f.shape[1] for f in features_list]

    print(f"Dataset: {args.dataset}")
    print(f"  Samples={num_samples}, Views={num_views}, Clusters={num_clusters}")
    print(f"  View dims: {view_dims}")

    data_module = MultiViewDataModule(
        features_list=features_list, labels=labels,
        batch_size=args.batch_size, num_workers=0,
    )
    full_loader = data_module.full_dataloader()

    # Complete baseline
    from evaluation.clustering_metrics import evaluate_model
    complete_metrics = evaluate_model(model, full_loader, device)
    print(f"\nComplete baseline: ACC={complete_metrics['ACC']:.4f} "
          f"NMI={complete_metrics['NMI']:.4f} ARI={complete_metrics['ARI']:.4f}")

    if args.protocol == 'all':
        protocols = ['completer', 'dcg_original', 'per_view_ind', 'cell_level']
    else:
        protocols = [args.protocol]

    all_results = {
        'complete': {k: float(v) for k, v in complete_metrics.items()},
        'protocols': {},
    }

    for proto in protocols:
        print(f"\n{'='*70}")
        print(f"Protocol: {proto}")
        print(f"{'='*70}")

        proto_results = run_single_protocol(
            model, full_loader, device,
            num_samples, num_views, proto,
            args.missing_rates, args.num_trials, args.seed,
            complete_metrics
        )
        all_results['protocols'][proto] = proto_results

    # Summary Table 1: grouped by (nominal mr x protocol)
    print(f"\n\n{'='*100}")
    print("SUMMARY Table 1: Performance grouped by (nominal mr, protocol)")
    print(f"{'='*100}")
    print(f"Dataset: {args.dataset}, Views={num_views}, "
          f"Complete ACC={complete_metrics['ACC']:.4f}")
    print()

    header = (f"{'mr':>6}  {'Protocol':<16}  {'Real cell MR':>13}  "
              f"{'Complete %':>11}  {'Avg Views':>10}  "
              f"{'ACC':>16}  {'NMI':>16}  {'ACC Drop':>10}")
    print(header)
    print("-" * len(header))

    for rate in args.missing_rates:
        rate_str = str(rate)
        for proto in protocols:
            if rate_str in all_results['protocols'].get(proto, {}):
                r = all_results['protocols'][proto][rate_str]
                cell_mr = r.get('cell_missing_rate', 0)
                complete_ratio = r.get('complete_sample_ratio', 0)
                avg_v = r.get('avg_views_per_sample', 0)
                acc_mean = r['ACC']['mean']
                acc_std = r['ACC']['std']
                nmi_mean = r['NMI']['mean']
                nmi_std = r['NMI']['std']
                drop = complete_metrics['ACC'] - acc_mean
                drop_pct = drop / complete_metrics['ACC'] * 100

                print(f"{rate:>6.1f}  {proto:<16}  {cell_mr:>12.2%}  "
                      f"{complete_ratio:>10.2%}  {avg_v:>10.2f}  "
                      f"{acc_mean:>7.4f}+/-{acc_std:.4f}  "
                      f"{nmi_mean:>7.4f}+/-{nmi_std:.4f}  "
                      f"{f'-{drop_pct:.1f}%':>10}")
        print()

    # Summary Table 2: sorted by real cell MR
    print(f"\n{'='*100}")
    print("SUMMARY Table 2: Sorted by REAL cell missing rate (for cross-method comparison)")
    print(f"{'='*100}")
    print()

    all_rows = []
    for rate in args.missing_rates:
        rate_str = str(rate)
        for proto in protocols:
            if rate_str in all_results['protocols'].get(proto, {}):
                r = all_results['protocols'][proto][rate_str]
                all_rows.append({
                    'nominal_mr': rate,
                    'protocol': proto,
                    'real_cell_mr': r.get('cell_missing_rate', 0),
                    'complete_ratio': r.get('complete_sample_ratio', 0),
                    'avg_views': r.get('avg_views_per_sample', 0),
                    'acc_mean': r['ACC']['mean'],
                    'acc_std': r['ACC']['std'],
                    'nmi_mean': r['NMI']['mean'],
                })

    all_rows.sort(key=lambda x: x['real_cell_mr'])

    header2 = (f"{'Real cell MR':>13}  {'Nominal mr':>11}  {'Protocol':<16}  "
               f"{'Complete %':>11}  {'Avg Views':>10}  "
               f"{'ACC':>16}  {'NMI':>9}")
    print(header2)
    print("-" * len(header2))

    for row in all_rows:
        print(f"{row['real_cell_mr']:>12.2%}  "
              f"{row['nominal_mr']:>11.1f}  "
              f"{row['protocol']:<16}  "
              f"{row['complete_ratio']:>10.2%}  "
              f"{row['avg_views']:>10.2f}  "
              f"{row['acc_mean']:>7.4f}+/-{row['acc_std']:.4f}  "
              f"{row['nmi_mean']:>9.4f}")

    # Save results
    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    os.makedirs(output_dir, exist_ok=True)

    all_results['config'] = {
        'checkpoint': args.checkpoint,
        'dataset': args.dataset,
        'num_views': num_views,
        'view_dims': view_dims,
        'num_samples': num_samples,
        'num_clusters': num_clusters,
        'missing_rates': args.missing_rates,
        'protocols': protocols,
        'num_trials': args.num_trials,
        'seed': args.seed,
        'selected_views': args.selected_views,
        'mask_strategy': args.mask_strategy,
    }

    results_path = os.path.join(output_dir, 'four_protocol_eval_results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(all_results, f, default_flow_style=False)

    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
