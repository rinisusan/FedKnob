"""Seed-stability sweep for the Week 3 household partition.

Why this exists
---------------
Every partition in the project -- households_{20,50,100,200}.parquet -- was drawn
with ``seed=42``. The JSD-vs-household-count relation reported in
The heterogeneity measurement study is therefore four design points at a
SINGLE draw each, and every federated result from Week 4 onward runs on one
particular realisation of ``households_100``.

That is an unquantified single point of failure:

  * a partition seed is UPSTREAM of everything -- an atypical draw biases every
    downstream ``adapter_gain`` and ASR number in a correlated direction, and no
    amount of downstream rigour can detect or undo it;
  * nothing inside the pipeline signals "this draw was lucky";
  * the Dirichlet draw genuinely is high-variance (60 intents each split across
    up to 200 bins at alpha=0.5), so the spread is plausibly large.

This script measures that spread. It re-draws the partition across several seeds
for each household count and reports mean +/- sd of the divergence metrics, so
the study can state intervals instead of point estimates.

It does NOT modify ``partition_by_speaker_dirichlet``: it imports and calls the
existing engine, and reuses the divergence maths from
``analyze_partition_divergence.py`` verbatim (imported, not re-implemented, so
the two can never silently drift apart).

Nothing is written to ``artifacts/partitions/`` -- the canonical seed-42 parquets
are left untouched. Label matrices are built in memory.

Pre-registration warning
------------------------
Choosing the canonical seed AFTER seeing this sweep would be p-hacking the
foundation of the whole programme. Commit to a selection rule BEFORE running:
``--rule median`` (take the median-JSD seed) or ``--rule first`` (keep 42
regardless, and use the sweep only for error bars). The rule you pass is recorded
in the report.

Usage
-----
    python scripts/sweep_partition_seeds.py
    python scripts/sweep_partition_seeds.py --seeds 42 43 44 45 46
    python scripts/sweep_partition_seeds.py --households 100 --seeds 42 43 44
    python scripts/sweep_partition_seeds.py --plot

Outputs
-------
    artifacts/partitions/seed_sweep.json   -- per-seed rows + aggregates
    reports/seed_sweep_jsd.png             -- mean +/- sd vs household count (--plot)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from fedknob.data.massive import PROJECT_ROOT, load_locale
from fedknob.data.partition import (
    DIRICHLET_ALPHA,
    household_label_matrix,
    partition_by_speaker_dirichlet,
)

# Reuse the EXACT divergence maths the canonical analysis uses. scripts/ is not a
# package, so put it on the path and import the module directly. Importing rather
# than copying is deliberate: duplicated JSD code would eventually disagree with
# analyze_partition_divergence.py and silently invalidate the comparison.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_partition_divergence import (  # noqa: E402
    js_divergence,
    to_distributions,
)

PARTITION_DIR = PROJECT_ROOT / "artifacts" / "partitions"
REPORT_DIR = PROJECT_ROOT / "reports"

# Size bands that reproduce the canonical seed-42 partitions. The band MUST scale
# with the household count: at 20 households the mean is ~575 utterances, so the
# module default of [30, 300] is arithmetically infeasible and the repair pass
# would thrash. These values are the ones the committed partitions were built
# with -- the --self-check flag verifies that by reproducing seed 42's JSD.
SIZE_BANDS: dict[int, tuple[int, int]] = {
    20: (300, 1200),
    50: (100, 500),
    100: (50, 300),
    200: (30, 300),
}

DEFAULT_HOUSEHOLDS = (20, 50, 100, 200)
DEFAULT_SEEDS = (42, 43, 44, 45, 46)


def size_band(n_households: int) -> tuple[int, int]:
    """Band for a household count; interpolate sensibly for unlisted counts."""
    if n_households in SIZE_BANDS:
        return SIZE_BANDS[n_households]
    # fall back to a proportional band around the mean household size
    mean = 11514 / n_households
    return max(1, int(0.5 * mean)), int(2.6 * mean)


# --------------------------------------------------------------------------- #
# Metrics for a single draw
# --------------------------------------------------------------------------- #
def divergence_stats(result, df) -> dict:
    """Divergence metrics for one PartitionResult, computed in memory."""
    counts, _names = household_label_matrix(result, df)
    counts = np.asarray(counts, dtype=float)
    dists = to_distributions(counts)

    global_dist = counts.sum(axis=0) / counts.sum()
    to_global = js_divergence(dists, np.broadcast_to(global_dist, dists.shape))

    iu = np.triu_indices(dists.shape[0], k=1)
    pair = js_divergence(dists[iu[0]], dists[iu[1]])

    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(dists > 0, dists * np.log(dists), 0.0), axis=1)
    eff = np.exp(ent)

    sizes = counts.sum(axis=1)
    return {
        "jsd_to_global_mean": float(to_global.mean()),
        "jsd_to_global_median": float(np.median(to_global)),
        "jsd_pairwise_mean": float(pair.mean()),
        "jsd_pairwise_median": float(np.median(pair)),
        "eff_intents_median": float(np.median(eff)),
        "utts_min": int(sizes.min()),
        "utts_median": int(np.median(sizes)),
        "utts_max": int(sizes.max()),
    }


def run_one(df, n_households: int, seed: int, alpha: float) -> dict:
    lo, hi = size_band(n_households)
    result = partition_by_speaker_dirichlet(
        df,
        num_households=n_households,
        alpha=alpha,
        seed=seed,
        min_utts=lo,
        max_utts=hi,
    )
    row = {"n_households": n_households, "seed": seed, "alpha": alpha,
           "min_utts_band": lo, "max_utts_band": hi}
    row.update(divergence_stats(result, df))
    return row


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
METRICS = ("jsd_to_global_mean", "jsd_pairwise_mean", "eff_intents_median",
           "utts_median")


def aggregate(rows: list[dict]) -> list[dict]:
    """mean / sd / min / max per household count. sd is the sample sd (ddof=1)."""
    out = []
    for n in sorted({r["n_households"] for r in rows}):
        grp = [r for r in rows if r["n_households"] == n]
        agg = {"n_households": n, "n_seeds": len(grp),
               "seeds": [r["seed"] for r in grp]}
        for m in METRICS:
            v = np.array([r[m] for r in grp], dtype=float)
            sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            agg[m] = {
                "mean": round(float(v.mean()), 4),
                "sd": round(sd, 4),
                "min": round(float(v.min()), 4),
                "max": round(float(v.max()), 4),
                "cv": round(sd / float(v.mean()), 4) if v.mean() else None,
            }
        out.append(agg)
    return out


def separation_check(agg: list[dict], metric: str = "jsd_to_global_mean") -> list[dict]:
    """Are adjacent household counts distinguishable ABOVE seed noise?

    This is the decision-relevant question. The heterogeneity study claims client
    count is the dominant control; that only holds if the gap between adjacent
    design points is large relative to the seed-to-seed spread. We use a
    two-sample-style separation ratio: gap / pooled sd. A ratio >> 2 means the
    design points are cleanly resolved; ~1 or below means the "scaling law" is
    partly measuring noise.
    """
    out = []
    for a, b in zip(agg, agg[1:], strict=False):
        gap = abs(b[metric]["mean"] - a[metric]["mean"])
        pooled = np.sqrt((a[metric]["sd"] ** 2 + b[metric]["sd"] ** 2) / 2)
        out.append({
            "pair": f"{a['n_households']} vs {b['n_households']}",
            "gap": round(gap, 4),
            "pooled_sd": round(float(pooled), 4),
            "separation": round(float(gap / pooled), 1) if pooled > 0 else None,
            "resolved": bool(pooled == 0 or gap / pooled > 2.0),
        })
    return out


def refit_scaling(rows: list[dict], n_speakers: int = 682) -> dict:
    """Refit JSD ~ a + b/sqrt(speakers_per_household) SEPARATELY PER SEED.

    The study reports a single fit (R^2=0.9989) from the seed-42 points. If the
    per-seed coefficients are tight, the relation is real; if they scatter, the
    headline fit was partly luck.
    """
    seeds = sorted({r["seed"] for r in rows})
    fits = []
    for s in seeds:
        grp = sorted((r for r in rows if r["seed"] == s),
                     key=lambda r: r["n_households"])
        if len(grp) < 3:
            continue
        x = np.array([1.0 / np.sqrt(n_speakers / r["n_households"]) for r in grp])
        y = np.array([r["jsd_to_global_mean"] for r in grp])
        b, a = np.polyfit(x, y, 1)
        pred = a + b * x
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        fits.append({"seed": s, "intercept": round(float(a), 4),
                     "slope": round(float(b), 4),
                     "r2": round(1 - ss_res / ss_tot, 4) if ss_tot else None})
    if not fits:
        return {"fits": [], "note": "need >=3 household counts per seed to fit"}
    return {
        "fits": fits,
        "slope_mean": round(float(np.mean([f["slope"] for f in fits])), 4),
        "slope_sd": round(float(np.std([f["slope"] for f in fits], ddof=1)), 4)
        if len(fits) > 1 else 0.0,
        "intercept_mean": round(float(np.mean([f["intercept"] for f in fits])), 4),
        "r2_min": min(f["r2"] for f in fits),
    }


def self_check(rows: list[dict]) -> list[str]:
    """Confirm the seed-42 rows reproduce the committed divergence_report.json.

    If these disagree, the size bands in SIZE_BANDS are not the ones the canonical
    partitions were built with, and the whole sweep is measuring a different
    object than the study reports. Fail loudly rather than quietly.
    """
    ref_path = PARTITION_DIR / "divergence_report.json"
    if not ref_path.exists():
        return [f"SKIP  no {ref_path.name} to check against"]
    ref = {r["n_households"]: r for r in json.loads(ref_path.read_text())}
    msgs = []
    for r in (x for x in rows if x["seed"] == 42):
        n = r["n_households"]
        if n not in ref:
            continue
        got, want = round(r["jsd_to_global_mean"], 4), ref[n]["jsd_to_global_mean"]
        ok = abs(got - want) < 5e-4
        msgs.append(f"{'PASS' if ok else 'FAIL'}  {n:>3} households: "
                    f"sweep={got:.4f}  committed={want:.4f}")
    return msgs


# --------------------------------------------------------------------------- #
def plot(agg: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = [a["n_households"] for a in agg]
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric, label, colour in (
        ("jsd_to_global_mean", "JSD to global", "tab:blue"),
        ("jsd_pairwise_mean", "pairwise JSD", "tab:orange"),
    ):
        mu = [a[metric]["mean"] for a in agg]
        sd = [a[metric]["sd"] for a in agg]
        ax.errorbar(n, mu, yerr=sd, marker="o", capsize=4, label=label, color=colour)
    ax.set_xscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([str(x) for x in n])
    ax.set_xlabel("households")
    ax.set_ylabel("Jensen-Shannon divergence (bits)")
    ax.set_title("Partition heterogeneity vs client count\n"
                 f"mean +/- sd over {agg[0]['n_seeds']} seeds")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Partition seed-stability sweep")
    ap.add_argument("--locale", default="en-US")
    ap.add_argument("--partition", default="train",
                    help="MASSIVE split (must match the canonical partitions)")
    ap.add_argument("--households", type=int, nargs="+", default=DEFAULT_HOUSEHOLDS)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--alpha", type=float, default=DIRICHLET_ALPHA)
    ap.add_argument("--rule", choices=("median", "first"), default="median",
                    help="PRE-COMMIT this before looking at results: 'median' "
                         "selects the median-JSD seed as canonical, 'first' keeps "
                         "seed 42 and uses the sweep only for error bars")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--no-self-check", action="store_true")
    args = ap.parse_args()

    part_arg = None if args.partition == "all" else args.partition
    df = load_locale(args.locale, partition=part_arg)
    print(f"Loaded MASSIVE {args.locale} ({args.partition}): {len(df)} utterances, "
          f"{df['worker_id'].nunique()} speakers, {df['intent'].nunique()} intents")
    print(f"Sweep: {len(args.households)} household counts x {len(args.seeds)} seeds "
          f"= {len(args.households) * len(args.seeds)} partitions "
          f"(alpha={args.alpha}, rule={args.rule})\n")

    rows = []
    for n in args.households:
        for s in args.seeds:
            r = run_one(df, n, s, args.alpha)
            rows.append(r)
            print(f"  n={n:<4} seed={s:<4} "
                  f"JSD->global={r['jsd_to_global_mean']:.4f}  "
                  f"pairwise={r['jsd_pairwise_mean']:.4f}  "
                  f"eff_int={r['eff_intents_median']:.1f}  "
                  f"utts={r['utts_min']}/{r['utts_median']}/{r['utts_max']}")

    agg = aggregate(rows)

    print("\n" + "=" * 78)
    print("MEAN +/- SD ACROSS SEEDS")
    print("=" * 78)
    hdr = f"{'hh':>4} {'seeds':>6} {'JSD->global':>18} {'pairwise JSD':>18} {'eff_int':>14}"
    print(hdr)
    print("-" * len(hdr))
    for a in agg:
        g, p, e = (a["jsd_to_global_mean"], a["jsd_pairwise_mean"],
                   a["eff_intents_median"])
        print(f"{a['n_households']:>4} {a['n_seeds']:>6} "
              f"{g['mean']:>10.4f} +/-{g['sd']:<6.4f} "
              f"{p['mean']:>10.4f} +/-{p['sd']:<6.4f} "
              f"{e['mean']:>7.1f} +/-{e['sd']:<5.2f}")

    print("\nRelative spread (cv = sd/mean) of JSD->global:")
    for a in agg:
        cv = a["jsd_to_global_mean"]["cv"]
        print(f"  {a['n_households']:>4} households: cv = {cv:.1%}" if cv is not None
              else f"  {a['n_households']:>4} households: cv = n/a")

    sep = separation_check(agg)
    if sep:
        print("\nAre adjacent design points resolved above seed noise?")
        for s in sep:
            print(f"  {s['pair']:<14} gap={s['gap']:.4f}  pooled_sd={s['pooled_sd']:.4f}"
                  f"  separation={s['separation']}x  "
                  f"{'RESOLVED' if s['resolved'] else '** NOT RESOLVED **'}")

    fit = refit_scaling(rows)
    if fit.get("fits"):
        print("\nPer-seed refit of JSD ~ a + b/sqrt(speakers per household):")
        print(f"  slope     = {fit['slope_mean']} +/- {fit['slope_sd']}")
        print(f"  intercept = {fit['intercept_mean']}")
        print(f"  worst per-seed R^2 = {fit['r2_min']}")

    checks = [] if args.no_self_check else self_check(rows)
    if checks:
        print("\nSelf-check vs committed divergence_report.json (seed 42):")
        for c in checks:
            print(f"  {c}")

    # selection rule -- applied to the household count used for Weeks 4-9
    canonical = {}
    for n in args.households:
        grp = sorted((r for r in rows if r["n_households"] == n),
                     key=lambda r: r["jsd_to_global_mean"])
        canonical[n] = (args.seeds[0] if args.rule == "first"
                        else grp[len(grp) // 2]["seed"])
    print(f"\nSelection rule '{args.rule}' -> canonical seed per household count:")
    for n, s in canonical.items():
        print(f"  {n:>4} households: seed {s}")
    print("\n  NOTE: this rule must have been chosen BEFORE running the sweep.")
    print("  Selecting a seed after inspecting the numbers invalidates every")
    print("  downstream result that uses the partition.")

    out = {
        "config": {"locale": args.locale, "partition": args.partition,
                   "alpha": args.alpha, "households": list(args.households),
                   "seeds": list(args.seeds), "rule": args.rule,
                   "size_bands": {str(n): list(size_band(n))
                                  for n in args.households}},
        "runs": rows,
        "aggregate": agg,
        "separation": sep,
        "scaling_refit": fit,
        "self_check": checks,
        "canonical_seed_by_household_count": canonical,
    }
    out_path = PARTITION_DIR / "seed_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nreport -> {out_path}")

    if args.plot:
        p = REPORT_DIR / "seed_sweep_jsd.png"
        plot(agg, p)
        print(f"plot   -> {p}")


if __name__ == "__main__":
    main()
