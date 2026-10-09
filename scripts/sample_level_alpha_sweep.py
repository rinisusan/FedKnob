"""Controlled contrast: does Dirichlet alpha lose its leverage at OWNER granularity?

The claim in Finding 1
----------------------
"Under speaker-atomic partitioning, alpha is inert." That is a *comparative* claim: it
presupposes a baseline in which alpha works. Currently that baseline is Hsu et al.
(arXiv:1909.06335), who report a 30.1% -> 76.9% accuracy swing across their alpha range on
CIFAR-10.

That comparison is confounded six ways: different corpus, modality, class count (10 vs 60),
samples per class (~5,000 vs ~192), client count, and -- worst -- a different *metric*
entirely. A 46-point accuracy swing and a 0.53-class entropy change are not comparable
quantities, so the contrast is currently rhetorical rather than quantitative.

What this script does
---------------------
Runs BOTH partitioning schemes on the SAME corpus, over the SAME alpha values and client
counts, scored with the SAME metric functions:

    sample-level   -- Dirichlet applied to individual utterances (Hsu et al.'s method)
    speaker-level  -- Dirichlet applied to whole speakers    (our method, unchanged)

Everything is held constant except the atomic unit. If alpha swings the metric at sample
level and not at speaker level, granularity is the only variable left that can explain it.

Possible outcomes, stated in advance
------------------------------------
    A. large sample-level span, ~zero speaker-level span
       -> Finding 1 confirmed AND isolated. The strong result.
    B. small span at BOTH levels
       -> Finding 1 is not about granularity. It would instead be about MASSIVE's 60-class
          structure, or about effective-intents being an insensitive metric. This would
          undermine the finding -- which is exactly why the experiment is worth running.
    C. moderate sample-level span
       -> a real ratio instead of a hand-wave; a more nuanced claim.

Nothing in ``fedknob.data.partition`` is modified. The metric functions are IMPORTED
from the existing modules rather than reimplemented, because the whole validity of this
comparison rests on both arms being scored by identical code.

Usage
-----
    python scripts/sample_level_alpha_sweep.py
    python scripts/sample_level_alpha_sweep.py --households 200 --match-sizes
    python scripts/sample_level_alpha_sweep.py --alphas 0.05 0.1 0.3 0.5 1.0 --seeds 42 43 44

Outputs
-------
    artifacts/partitions/granularity_contrast.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fedknob.data.massive import PROJECT_ROOT, load_locale
from fedknob.data.partition import (
    _dirichlet_split_indices,
    _effective_intents,
    partition_by_speaker_dirichlet,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_partition_divergence import js_divergence, to_distributions  # noqa: E402

PARTITION_DIR = PROJECT_ROOT / "artifacts" / "partitions"

# Same bands the canonical speaker-level partitions use; applied to the sample-level arm
# only when --match-sizes is set.
SIZE_BANDS: dict[int, tuple[int, int]] = {
    20: (300, 1200), 50: (100, 500), 100: (50, 300), 200: (30, 300),
}
DEFAULT_ALPHAS = (0.05, 0.1, 0.3, 0.5, 1.0)
DEFAULT_HOUSEHOLDS = (20, 200)
DEFAULT_SEEDS = (42,)


def band(n: int) -> tuple[int, int]:
    if n in SIZE_BANDS:
        return SIZE_BANDS[n]
    mean = 11514 / n
    return max(1, int(0.5 * mean)), int(2.6 * mean)


# --------------------------------------------------------------------------- #
# Sample-level partitioning -- Hsu et al.'s construction, applied to utterances
# --------------------------------------------------------------------------- #
def partition_by_sample_dirichlet(
    df: pd.DataFrame,
    n_clients: int,
    alpha: float,
    seed: int,
    min_utts: int | None = None,
    max_utts: int | None = None,
) -> dict[int, list[int]]:
    """Dirichlet label skew at UTTERANCE granularity. ``worker_id`` is ignored.

    For each intent c: draw p_c ~ Dir(alpha * 1_N) and split that intent's rows across
    clients following p_c. Identical in structure to the speaker-level routine except that
    the unit being distributed is a single row rather than a whole speaker -- which is the
    one variable this experiment isolates.

    ``_dirichlet_split_indices`` is imported rather than reimplemented so that both arms
    use the same largest-remainder apportionment.
    """
    rng = np.random.default_rng(seed)
    assignment: dict[int, list[int]] = {h: [] for h in range(n_clients)}

    for intent in sorted(df["intent"].unique()):
        rows = sorted(int(i) for i in np.where(df["intent"].to_numpy() == intent)[0])
        proportions = rng.dirichlet(np.full(n_clients, alpha))
        bins = _dirichlet_split_indices(len(rows), proportions)
        perm = rng.permutation(n_clients)
        for r, b in zip(rows, bins, strict=True):
            assignment[int(perm[b])].append(r)

    if min_utts is not None and max_utts is not None:
        assignment = _repair_rows(assignment, min_utts, max_utts, rng)
    return {h: sorted(v) for h, v in assignment.items()}


def _repair_rows(assignment, min_utts, max_utts, rng):
    """Move individual utterances from oversized to undersized clients.

    Trivial at sample level -- there is no atomicity constraint to respect. Used only for
    the --match-sizes arm, so that a size difference cannot be offered as an alternative
    explanation for a metric difference.
    """
    for _ in range(10_000):
        sizes = {h: len(v) for h, v in assignment.items()}
        lo = min(sizes, key=lambda h: sizes[h])
        hi = max(sizes, key=lambda h: sizes[h])
        if sizes[lo] >= min_utts and sizes[hi] <= max_utts:
            break
        if sizes[hi] - 1 < min_utts:
            break
        j = int(rng.integers(len(assignment[hi])))
        assignment[lo].append(assignment[hi].pop(j))
    return assignment


# --------------------------------------------------------------------------- #
# Scoring -- identical for both arms
# --------------------------------------------------------------------------- #
def score(assignment: dict[int, list[int]], df: pd.DataFrame) -> dict:
    """Effective intents + JSD->global. Same metric code path for both arms."""
    intents = sorted(df["intent"].unique())
    idx = {c: i for i, c in enumerate(intents)}
    counts = np.zeros((len(assignment), len(intents)), dtype=float)
    for h, rows in sorted(assignment.items()):
        if not rows:
            continue
        vc = df.iloc[rows]["intent"].value_counts()
        for c, n in vc.items():
            counts[h, idx[c]] = n

    eff = np.array([
        _effective_intents(df.iloc[rows]["intent"]) if rows else 0.0
        for _h, rows in sorted(assignment.items())
    ])
    dists = to_distributions(counts)
    gl = counts.sum(axis=0) / counts.sum()
    to_global = js_divergence(dists, np.broadcast_to(gl, dists.shape))
    n_distinct = np.array([(counts[h] > 0).sum() for h in range(len(assignment))])
    sizes = counts.sum(axis=1)

    return {
        "eff_min": round(float(eff.min()), 2),
        "eff_median": round(float(np.median(eff)), 2),
        "eff_max": round(float(eff.max()), 2),
        "distinct_median": int(np.median(n_distinct)),
        "jsd_to_global_mean": round(float(to_global.mean()), 4),
        "utts_min": int(sizes.min()),
        "utts_median": int(np.median(sizes)),
        "utts_max": int(sizes.max()),
    }


def run_grid(df, n_clients, alphas, seeds, match_sizes) -> list[dict]:
    lo, hi = band(n_clients)
    rows = []
    for a in alphas:
        for s in seeds:
            samp = partition_by_sample_dirichlet(
                df, n_clients, a, s,
                min_utts=lo if match_sizes else None,
                max_utts=hi if match_sizes else None,
            )
            spk = partition_by_speaker_dirichlet(
                df, num_households=n_clients, alpha=a, seed=s,
                min_utts=lo, max_utts=hi,
            )
            rows.append({
                "n_clients": n_clients, "alpha": a, "seed": s,
                "sample_level": score(samp, df),
                "speaker_level": score(spk.assignment, df),
            })
    return rows


def alpha_spans(rows: list[dict], arm: str, key: str = "eff_median") -> list[float]:
    """Span of the metric across the alpha sweep, computed WITHIN each seed.

    Pooling seeds before taking max-min conflates the alpha effect with seed-to-seed
    variation and inflates the span. That was a real bug in an earlier version of this
    script: at N=200 it reported a speaker-level span of 1.73 where the correct
    per-seed mean is 0.98. Always reduce within a seed first, then aggregate.
    """
    out = []
    for s in sorted({r["seed"] for r in rows}):
        v = [r[arm][key] for r in rows if r["seed"] == s]
        if v:
            out.append(round(max(v) - min(v), 3))
    return out


def summarise(per_seed: list[float]) -> tuple[float, float]:
    """mean, sd (ddof=1; 0.0 for a single seed)."""
    m = float(np.mean(per_seed))
    sd = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0
    return round(m, 3), round(sd, 3)


def seed_noise_sd(n_clients: int) -> float | None:
    """Effective-intent seed sd for this client count, from the partition seed sweep.

    Loaded rather than hardcoded so the 'is this span bigger than noise?' comparison
    cannot drift out of sync with seed_sweep.json.
    """
    p = PARTITION_DIR / "seed_sweep.json"
    if not p.exists():
        return None
    try:
        agg = json.loads(p.read_text())["aggregate"]
        for a in agg:
            if a["n_households"] == n_clients:
                return float(a["eff_intents_median"]["sd"])
    except (KeyError, ValueError, TypeError):
        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample- vs speaker-level alpha leverage")
    ap.add_argument("--locale", default="en-US")
    ap.add_argument("--partition", default="train")
    ap.add_argument("--households", type=int, nargs="+", default=DEFAULT_HOUSEHOLDS)
    ap.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--match-sizes", action="store_true",
                    help="constrain the sample-level arm to the same size band as the "
                         "speaker-level arm, so a size difference cannot be offered as an "
                         "alternative explanation for a metric difference")
    args = ap.parse_args()

    part = None if args.partition == "all" else args.partition
    df = load_locale(args.locale, partition=part).reset_index(drop=True)
    print(f"Loaded MASSIVE {args.locale} ({args.partition}): {len(df)} utterances, "
          f"{df['worker_id'].nunique()} speakers, {df['intent'].nunique()} intents")
    print(f"alphas={list(args.alphas)}  N={list(args.households)}  seeds={list(args.seeds)}"
          f"  match_sizes={args.match_sizes}\n")

    out = {"config": vars(args), "grids": {}, "summary": []}

    for n in args.households:
        rows = run_grid(df, n, args.alphas, args.seeds, args.match_sizes)
        out["grids"][str(n)] = rows

        print("=" * 84)
        print(f"N = {n} clients")
        print("=" * 84)
        hdr = (f"{'alpha':>6} | {'SAMPLE-level eff (min/med/max)':>32} | "
               f"{'SPEAKER-level eff (min/med/max)':>32}")
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            a, sa, sp = r["alpha"], r["sample_level"], r["speaker_level"]
            print(f"{a:>6} | {sa['eff_min']:>9.2f} /{sa['eff_median']:>8.2f} /"
                  f"{sa['eff_max']:>9.2f}   | {sp['eff_min']:>9.2f} /"
                  f"{sp['eff_median']:>8.2f} /{sp['eff_max']:>9.2f}")

        s_per = alpha_spans(rows, "sample_level")
        k_per = alpha_spans(rows, "speaker_level")
        s_span, s_sd = summarise(s_per)
        k_span, k_sd = summarise(k_per)
        ratio = round(s_span / k_span, 1) if k_span > 0 else None
        print("\n  median effective-intent span across the alpha sweep")
        print("    (span computed WITHIN each seed, then averaged across seeds)")
        print(f"    sample-level (Hsu et al. method) : {s_span} +/- {s_sd}   per-seed {s_per}")
        print(f"    speaker-level (ours)             : {k_span} +/- {k_sd}   per-seed {k_per}")
        print(f"    leverage ratio                   : {ratio}x")
        noise = seed_noise_sd(n)
        if noise:
            print(f"    speaker span vs seed noise       : {k_span}/{noise} = "
                  f"{k_span / noise:.1f}x seed sd"
                  f"{'  (WITHIN NOISE)' if k_span <= noise else ''}")
        sj, _ = summarise(alpha_spans(rows, "sample_level", "jsd_to_global_mean"))
        kj, _ = summarise(alpha_spans(rows, "speaker_level", "jsd_to_global_mean"))
        print("\n  JSD->global span")
        print(f"    sample-level  : {sj}")
        print(f"    speaker-level : {kj}"
              + (f"   (ratio {sj / kj:.1f}x)" if kj else ""))
        sz = rows[0]
        print(f"\n  client sizes (alpha={rows[0]['alpha']}) "
              f"sample={sz['sample_level']['utts_min']}/"
              f"{sz['sample_level']['utts_median']}/{sz['sample_level']['utts_max']}  "
              f"speaker={sz['speaker_level']['utts_min']}/"
              f"{sz['speaker_level']['utts_median']}/{sz['speaker_level']['utts_max']}")
        if not args.match_sizes:
            print("  (sizes are unconstrained in the sample arm; rerun with --match-sizes\n"
                  "   to rule out volume as an alternative explanation)")
        print()

        out["summary"].append({
            "n_clients": n,
            "sample_eff_span": s_span, "sample_eff_span_sd": s_sd,
            "sample_eff_span_per_seed": s_per,
            "speaker_eff_span": k_span, "speaker_eff_span_sd": k_sd,
            "speaker_eff_span_per_seed": k_per,
            "leverage_ratio": ratio,
            "sample_jsd_span": sj, "speaker_jsd_span": kj,
            "seed_noise_sd": noise,
            "speaker_span_over_seed_noise": round(k_span / noise, 2) if noise else None,
        })

    # Two floors, both reported, because they answer different questions and the
    # write-up quotes the second:
    #
    #   per-N    each client count against its own seed sd -- the tighter test
    #   pooled   every client count against one floor, the mean of the per-N sds
    #
    # The figure draws a single dashed line, so it has to be the pooled floor; the
    # paper's "5.8x at N=20, 3.2x at N=200" is read against that same line. Without
    # this field a reader comparing the artifact with the paper finds 5.11 and 3.65
    # and has no way to see that the two are measuring against different floors.
    floors = [r["seed_noise_sd"] for r in out["summary"] if r["seed_noise_sd"]]
    if floors:
        pooled = sum(floors) / len(floors)
        out["pooled_seed_noise_sd"] = round(pooled, 4)
        for r in out["summary"]:
            r["speaker_span_over_pooled_seed_noise"] = round(r["speaker_eff_span"] / pooled, 2)

    print("=" * 84)
    print("VERDICT")
    print("=" * 84)
    for s in out["summary"]:
        r = s["leverage_ratio"]
        if r is None:
            verdict = "speaker-level span is zero -- alpha has no measurable effect"
        elif r >= 5:
            verdict = "OUTCOME A -- alpha's leverage collapses at owner granularity"
        elif r >= 2:
            verdict = "OUTCOME C -- reduced but non-trivial leverage"
        else:
            verdict = ("OUTCOME B -- alpha is weak at BOTH levels; the inertness is NOT "
                       "about granularity")
        print(f"  N={s['n_clients']:>4}: ratio {r}x  ->  {verdict}")
    print("\n  Compare each span against the seed-noise sd for that N "
          "(seed_sweep.json):\n  a span smaller than the seed sd is not an effect.")

    p = PARTITION_DIR / "granularity_contrast.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nreport -> {p}")


if __name__ == "__main__":
    main()
