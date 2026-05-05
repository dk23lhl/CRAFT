"""Compare mask-strategy IMVC results across {learnable, mean, attention_mask, zero}.

Usage:
    python scripts/compare_strategies.py hw cub                          # default prefix=C3_best
    python scripts/compare_strategies.py --prefix C4_hw_mft hw           # MFT ckpts
    python scripts/compare_strategies.py --prefix C4_hw_mft \
        --suffix _seedmatched hw                                          # read eval_*_seedmatched/

Expected subdirectory layout (under each ckpt directory):
    eval_learnable{SUFFIX}/four_protocol_eval_results.yaml
    eval_mean{SUFFIX}/four_protocol_eval_results.yaml
    eval_attention_mask{SUFFIX}/four_protocol_eval_results.yaml
    eval_zero{SUFFIX}/four_protocol_eval_results.yaml

Note: the dataset suffix (_hw / _cub) is appended to the prefix automatically; if the prefix
already contains a dataset tag (e.g. C4_hw_mft), the dataset is not appended and the prefix is
globbed directly.

Read-only on yaml; does not modify any file.
"""

import argparse
import glob
import statistics as stats

import yaml

def make_strategies(suffix: str = ""):
    return [
        ("learnable",      f"eval_learnable{suffix}"),
        ("mean",           f"eval_mean{suffix}"),
        ("attention_mask", f"eval_attention_mask{suffix}"),
        ("zero",           f"eval_zero{suffix}"),
    ]

# 4 protocols x 4 missing rates = 16 cells, all output.
ALL_PROTOCOLS = ["completer", "dcg_original", "per_view_ind", "cell_level"]
ALL_MISSING_RATES = ["0.1", "0.3", "0.5", "0.7"]


def summarize(files, proto, mr):
    vs = []
    for f in files:
        try:
            r = yaml.safe_load(open(f))
            vs.append(r["protocols"][proto][mr]["ACC"]["mean"])
        except (KeyError, TypeError):
            pass
    if not vs:
        return None
    m = stats.mean(vs)
    s = stats.stdev(vs) if len(vs) > 1 else 0.0
    return m, s, len(vs)


def build_glob_pattern(prefix: str, ds: str, sub: str) -> str:
    """If prefix already contains _hw / _cub, don't append the dataset suffix; otherwise append it.

    After base, `*_s*` matches the seed segment, which covers both:
      C3_best_hw_s42                (base=C3_best_hw, empty middle)
      C4_hw_mft_p05_s42             (base=C4_hw_mft, middle is _p05)
    """
    if f"_{ds}" in prefix or prefix.endswith(f"_{ds}"):
        base = prefix
    else:
        base = f"{prefix}_{ds}"
    return f"outputs/CRAFT_MVC/{base}*_s*/{sub}/four_protocol_eval_results.yaml"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("datasets", nargs="*", default=["hw", "cub"],
                   help="dataset tag(s), e.g. hw cub (default: both)")
    p.add_argument("--prefix", default="C3_best",
                   help="exp_name prefix (default: C3_best)")
    p.add_argument("--suffix", default="",
                   help="subdirectory suffix, e.g. '_seedmatched' reads eval_*_seedmatched/ (default: empty)")
    args = p.parse_args()

    strategies = make_strategies(args.suffix)

    for ds in args.datasets:
        title = f"{ds.upper()}  (prefix={args.prefix}"
        if args.suffix:
            title += f", suffix={args.suffix}"
        title += ")"
        print(f"\n=========== {title} ===========")

        # complete-data baseline (each strategy's yaml also has a "complete" block)
        for label, sub in strategies:
            pat = build_glob_pattern(args.prefix, ds, sub)
            files = sorted(glob.glob(pat))
            if not files:
                print(f"  [skip] strategy={label}  (no files in {sub}/)")
                continue
            print(f"\n  --- strategy: {label}  (n={len(files)} seeds, subdir={sub}/) ---")

            comp_vs = []
            for f in files:
                try:
                    comp_vs.append(yaml.safe_load(open(f))["complete"]["ACC"])
                except (KeyError, TypeError):
                    pass
            if comp_vs:
                m = stats.mean(comp_vs)
                s = stats.stdev(comp_vs) if len(comp_vs) > 1 else 0.0
                print(f"    complete             : {m:.4f} +/- {s:.4f}")

            # All 4 protocols x 4 mr = 16 cells, output grouped by protocol.
            for proto in ALL_PROTOCOLS:
                print(f"    [{proto}]")
                for mr in ALL_MISSING_RATES:
                    res = summarize(files, proto, mr)
                    if res is None:
                        print(f"      mr={mr}: no data")
                    else:
                        m, s, n = res
                        print(f"      mr={mr}: {m:.4f} +/- {s:.4f}  (n={n})")


if __name__ == "__main__":
    main()
