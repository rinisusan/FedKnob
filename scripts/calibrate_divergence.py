"""Calibrate JSD-to-global against a size-matched null, per partition.

    python scripts/calibrate_divergence.py
    python scripts/calibrate_divergence.py --draws 50 --seed 7

Writes ``artifacts/partitions/calibrated_divergence.json``.

Why this script exists
----------------------
``analyze_partition_divergence.py`` reports raw mean JSD to the global label
distribution: 0.051 at N=20 rising to 0.326 at N=200. Read as a heterogeneity
measure, that says the partition becomes 6.4x more non-IID as clients narrow.

That reading is wrong, and this script is what shows it. JSD estimated from a
finite sample is biased *upward*: a client holding 55 utterances cannot cover 60
intents, so its empirical distribution looks divergent from global even when it
was drawn i.i.d. from global. The bias grows as clients shrink -- in the same
direction, over the same range, as the effect being claimed. Raw JSD therefore
cannot distinguish "these clients are genuinely different" from "these clients
are small".

The fix is a null that inherits the same sizes. For each client of size n we draw
n samples i.i.d. from the pooled distribution and recompute. The finite-sample
bias then appears in the floor as well as the measurement, and cancels in the
ratio. **Report the ratio, not the raw value.**

Doing this reverses the trend: the calibrated ratio *falls* monotonically as
clients narrow, because the floor rises faster than the signal.

Note on an earlier error
------------------------
A previous version of the study used a single null (0.0217) computed once at one
client size and applied it to all four partitions. That is size-matched only for
the coarsest partition and increasingly wrong for the rest -- it inflated the
N=200 ratio by roughly 10x and produced a rising trend where the correct
calculation gives a falling one. The null must be recomputed per partition, which
is what this script does.

Reads the partition parquet directly (household_id, intent), so it needs neither
the MASSIVE download nor a re-partition. Metric code is imported from
``analyze_partition_divergence`` so both arms are scored identically.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_partition_divergence import (  # noqa: E402
    js_divergence,
    label_matrix_from_parquet,
    to_distributions,
)

from fedknob.data.massive import PROJECT_ROOT  # noqa: E402

PART = PROJECT_ROOT / "artifacts" / "partitions"
OUT = PART / "calibrated_divergence.json"
DEFAULT_N = (20, 50, 100, 200)


def analyse(path: Path, draws: int, rng: np.random.Generator) -> dict:
    """Observed vs size-matched-null mean JSD to global for one partition file."""
    counts, _ = label_matrix_from_parquet(path)     # (clients x intents) COUNTS
    sizes = counts.sum(axis=1)
    gp = counts.sum(axis=0)
    gp = gp / gp.sum()
    # Same global reference and the same metric code the raw report uses, so the
    # observed column here must reproduce analyze_partition_divergence.py exactly.
    gp_b = np.broadcast_to(gp, counts.shape)

    observed = js_divergence(to_distributions(counts), gp_b)

    null = np.empty((draws, len(sizes)))
    for d in range(draws):
        sim = np.stack([
            np.bincount(rng.choice(len(gp), size=int(n), p=gp), minlength=len(gp))
            for n in sizes
        ]).astype(float)
        null[d] = js_divergence(to_distributions(sim), gp_b)
    null_mean = null.mean(axis=0)

    obs_m, null_m = float(observed.mean()), float(null_mean.mean())
    return {
        "n_clients": int(len(sizes)),
        "utts_median": float(np.median(sizes)),
        "utts_min": int(sizes.min()),
        "utts_max": int(sizes.max()),
        "jsd_observed_mean": round(obs_m, 4),
        "jsd_null_mean": round(null_m, 4),
        "ratio": round(obs_m / null_m, 3),
        "excess_absolute": round(obs_m - null_m, 4),
        # what fraction of the raw number is finite-sample artifact rather than skew
        "null_share_of_raw": round(null_m / obs_m, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="size-matched null calibration for JSD")
    ap.add_argument("--clients", type=int, nargs="*", default=list(DEFAULT_N))
    ap.add_argument("--draws", type=int, default=30,
                    help="null resamples per client; raise for tighter estimates")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    hdr = (f"{'N':>5} {'med utts':>9} {'raw JSD':>9} {'null':>8} {'ratio':>7} "
           f"{'null/raw':>9}")
    print(f"size-matched null: {args.draws} resamples per client, seed {args.seed}\n")
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for n in args.clients:
        p = PART / f"households_{n}.parquet"
        if not p.exists():
            print(f"{n:>5}  (missing {p.name})")
            continue
        r = analyse(p, args.draws, rng)
        results[str(n)] = r
        print(f"{n:>5} {r['utts_median']:>9.0f} {r['jsd_observed_mean']:>9.4f} "
              f"{r['jsd_null_mean']:>8.4f} {r['ratio']:>6.2f}x {r['null_share_of_raw']:>8.0%}")

    print("\nratio ~1.0  -> partition indistinguishable from i.i.d. sampling at these sizes")
    print("ratio >>1.0 -> genuine label skew beyond what small samples explain")
    if len(results) > 1:
        ks = sorted(results, key=int)
        first, last = results[ks[0]], results[ks[-1]]
        raw_dir = "rises" if last["jsd_observed_mean"] > first["jsd_observed_mean"] else "falls"
        cal_dir = "rises" if last["ratio"] > first["ratio"] else "falls"
        print(f"\nraw JSD {raw_dir} {first['jsd_observed_mean']:.4f} -> "
              f"{last['jsd_observed_mean']:.4f} across N={ks[0]}..{ks[-1]}")
        print(f"calibrated ratio {cal_dir} {first['ratio']:.2f}x -> {last['ratio']:.2f}x")
        if raw_dir != cal_dir:
            print("=> the raw statistic reports the OPPOSITE trend to the calibrated one.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "partitions": results}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
