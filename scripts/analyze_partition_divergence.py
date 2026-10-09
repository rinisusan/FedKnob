"""Measure how much heterogeneity a partition actually contains.

Why this exists
---------------
`verify_partition` reports **effective intents** = exp(entropy) of a household's
own label distribution. That answers "how concentrated is this household?", which
is the honest reading of the plan's "5-20 intents" target.

It does NOT answer the question Week 4's exit criterion depends on:

    is there enough heterogeneity for a per-client adapter to beat the global one?

Two households can each use 38 effective intents and still be near-identical to
each other (nothing to personalise on), or use the *same* 38 in very different
proportions (lots to personalise on). Effective intents cannot distinguish those.

The quantities that do
----------------------
1. **JSD(household, global)** -- the direct predictor of `adapter_gain`.
   If a household's label distribution equals the global one, its optimal adapter
   IS the global adapter, so `adapter_gain -> 0` no matter how good the code is.
   Mean JSD-to-global is therefore an upper-bound sanity check on Week 4's
   headline statistic BEFORE any training is run.

2. **pairwise JSD(h_i, h_j)** -- predicts per-client accuracy *spread* and, later,
   how separable clients are in Layer-1 parameter space.

Jensen-Shannon divergence is used (base-2, so it lands in [0, 1]) because it is
symmetric, finite even when supports differ, and bounded -- unlike KL, which
diverges when a household has zero mass on an intent (common here).

Usage
-----
    python scripts/analyze_partition_divergence.py                # all partitions found
    python scripts/analyze_partition_divergence.py --files a.parquet b.parquet
    python scripts/analyze_partition_divergence.py --heatmap      # also write figures

Reads the partition parquet directly (household_id, intent), so it needs neither
the MASSIVE download nor a re-partition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fedknob.data.massive import PROJECT_ROOT

PARTITION_DIR = PROJECT_ROOT / "artifacts" / "partitions"
REPORT_DIR = PROJECT_ROOT / "reports"


# --------------------------------------------------------------------------- #
# Divergence maths
# --------------------------------------------------------------------------- #
def _kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Row-wise KL(p||q) in bits; 0*log0 := 0."""
    mask = p > 0
    out = np.zeros_like(p, dtype=float)
    out[mask] = p[mask] * np.log2(p[mask] / q[mask])
    return out.sum(axis=-1)


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence in bits, in [0, 1]. Broadcasts over rows."""
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def label_matrix_from_parquet(path: Path) -> tuple[np.ndarray, list[str]]:
    """(households x intents) COUNT matrix straight from the partition file."""
    df = pd.read_parquet(path, engine="pyarrow")
    tab = (df.groupby(["household_id", "intent"]).size()
             .unstack(fill_value=0)
             .sort_index())
    return tab.to_numpy(dtype=float), list(tab.columns)


def to_distributions(counts: np.ndarray) -> np.ndarray:
    """Row-normalise counts -> per-household label distributions."""
    totals = counts.sum(axis=1, keepdims=True)
    return counts / np.clip(totals, 1, None)


def analyse(path: Path) -> dict:
    counts, intents = label_matrix_from_parquet(path)
    n_households, n_intents = counts.shape
    dists = to_distributions(counts)

    # global distribution = pooled counts (what the aggregated model fits)
    global_dist = counts.sum(axis=0) / counts.sum()

    # 1. JSD to global -- predicts adapter_gain
    to_global = js_divergence(dists, np.broadcast_to(global_dist, dists.shape))

    # 2. pairwise JSD -- predicts inter-client spread
    iu = np.triu_indices(n_households, k=1)
    pair = js_divergence(dists[iu[0]], dists[iu[1]])

    # effective intents, for cross-reference with verify_partition
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(dists > 0, dists * np.log(dists), 0.0), axis=1)
    eff = np.exp(ent)

    return {
        "file": path.name,
        "n_households": int(n_households),
        "n_intents": int(n_intents),
        "utts_total": int(counts.sum()),
        "utts_per_hh_median": float(np.median(counts.sum(axis=1))),
        "eff_intents_median": round(float(np.median(eff)), 2),
        # the decision metrics
        "jsd_to_global_mean": round(float(to_global.mean()), 4),
        "jsd_to_global_median": round(float(np.median(to_global)), 4),
        "jsd_to_global_max": round(float(to_global.max()), 4),
        "jsd_pairwise_mean": round(float(pair.mean()), 4),
        "jsd_pairwise_median": round(float(np.median(pair)), 4),
        "jsd_pairwise_p95": round(float(np.percentile(pair, 95)), 4),
        "_dists": dists,          # kept for the optional heatmap
    }


# --------------------------------------------------------------------------- #
def plot_pairwise(res: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = res["_dists"]
    n = d.shape[0]
    m = np.zeros((n, n))
    for i in range(n):
        m[i] = js_divergence(np.broadcast_to(d[i], d.shape), d)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(m, cmap="magma", interpolation="nearest")
    ax.set_title(f"Pairwise Jensen-Shannon divergence\n{res['file']} "
                 f"({res['n_households']} households)\n"
                 f"mean={res['jsd_pairwise_mean']:.3f} bits "
                 f"(0 = identical, 1 = disjoint)")
    ax.set_xlabel("household")
    ax.set_ylabel("household")
    fig.colorbar(im, ax=ax, label="JSD (bits)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Partition heterogeneity analysis")
    ap.add_argument("--files", nargs="+", default=None,
                    help="partition parquet paths (default: all households_*.parquet)")
    ap.add_argument("--heatmap", action="store_true",
                    help="also write a pairwise-JSD heatmap per partition")
    args = ap.parse_args()

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        paths = sorted(PARTITION_DIR.glob("households_*.parquet"),
                       key=lambda p: len(p.stem))
        paths = [p for p in paths if "__verify" not in p.stem]
    if not paths:
        raise SystemExit(f"no partition files found in {PARTITION_DIR}")

    results = [analyse(p) for p in paths]

    hdr = (f"{'partition':<22} {'hh':>4} {'utts/hh':>8} {'eff_int':>8} "
           f"{'JSD→global':>11} {'JSD pairwise':>13}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['file']:<22} {r['n_households']:>4} "
              f"{r['utts_per_hh_median']:>8.0f} {r['eff_intents_median']:>8.1f} "
              f"{r['jsd_to_global_mean']:>11.4f} {r['jsd_pairwise_mean']:>13.4f}")

    print("\nInterpretation")
    print("  JSD is in bits, 0 = identical distributions, 1 = disjoint supports.")
    print("  JSD→global  : how far each household is from the pooled distribution.")
    print("                This upper-bounds how much a personal adapter can beat")
    print("                the global one -- if it is ~0, adapter_gain will be ~0")
    print("                REGARDLESS of the federated code being correct.")
    print("  JSD pairwise: how different households are from each other; predicts")
    print("                per-client accuracy spread.")

    if len(results) > 1:
        base = max(results, key=lambda r: r["n_households"])
        print(f"\nRelative to {base['file']} (most heterogeneous reference):")
        for r in results:
            if r is base:
                continue
            g = r["jsd_to_global_mean"] / base["jsd_to_global_mean"]
            p = r["jsd_pairwise_mean"] / base["jsd_pairwise_mean"]
            print(f"  {r['file']:<22} retains {g:5.1%} of JSD→global, "
                  f"{p:5.1%} of pairwise JSD")

    out = PARTITION_DIR / "divergence_report.json"
    out.write_text(json.dumps(
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in results],
        indent=2))
    print(f"\nreport -> {out}")

    if args.heatmap:
        for r in results:
            p = REPORT_DIR / f"{Path(r['file']).stem}_pairwise_jsd.png"
            plot_pairwise(r, p)
            print(f"heatmap -> {p}")


if __name__ == "__main__":
    main()
