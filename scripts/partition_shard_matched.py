"""Sample-level Dirichlet partition, size-matched to a natural one.

    python scripts/partition_shard_matched.py
    python scripts/partition_shard_matched.py --alpha 0.1 --tag shard383_a01

The control arm for "does partition realism change what we measure?". It takes an
existing speaker-identity partition, keeps its **rows and its per-client sizes
exactly**, and re-assigns which rows land together using Dirichlet(alpha) over
intents -- ignoring who spoke them.

WHY SIZES ARE COPIED RATHER THAN DRAWN
--------------------------------------
Jensen-Shannon divergence between a client and the global pool is dominated by
client *size*, not label skew: a 26-utterance client cannot cover 60 intents, so
it looks divergent even when its rows were drawn at random. That apparent
divergence is the size-matched **floor**, and for this size distribution it is
0.3865 against a raw 0.5010 -- most of the measured heterogeneity is small
samples, not skew.

Copying the size vector means both arms share that floor exactly. Any difference
in raw JSD is then attributable to *composition* alone, which is the only thing
this partition is meant to change. Let the sizes drift and the comparison becomes
uninterpretable.

WHAT THIS IS FOR
----------------
Real speakers are narrow but narrow in *different* directions; Dirichlet shards
are narrow in ways that overlap. At 26 utterances a real speaker covers ~11.4
effective intents and an alpha=0.5 shard covers ~6.2. That gap was measured on
the *partition*; running both through FL asks
whether it changes any **outcome** -- which matters because essentially the whole
the FL literature evaluates on synthetic shards.

DETERMINISM
-----------
Seeded throughout and the row order is sorted before assignment, so the output is
byte-identical across runs and machines for a given (source, alpha, seed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedknob.data.massive import PROJECT_ROOT  # noqa: E402

OUT_DIR = PROJECT_ROOT / "artifacts" / "partitions"
DEFAULT_SOURCE = "artifacts/partitions/households_identity_min10.parquet"
REQUIRED = ("household_id", "row_index", "worker_id", "intent", "utt")


def effective_intents(counts) -> float:
    """exp(Shannon entropy) -- same statistic the other partition tools report."""
    c = np.asarray(list(counts), dtype=float)
    if c.sum() <= 0:
        return 0.0
    p = c[c > 0] / c.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def assign(sizes: np.ndarray, by_intent: dict[str, list[int]], alpha: float, seed: int):
    """Row indices per client. Dirichlet over intents, then fill to the exact size.

    Two passes, and the second is not optional. The Dirichlet draw is capped by
    what is actually left in each intent's pool, so a client whose draw asks for
    more of an intent than remains comes up short. Without a top-up the partition
    would hold fewer rows than the source and the two arms would stop being
    comparable -- the probe run of this logic lost ~1% of rows that way.

    Largest clients are served first: they have the most to place, and leaving
    them to last would force them to absorb whatever the others rejected, which
    reads as skew the Dirichlet never produced.
    """
    rng = np.random.default_rng(seed)
    intents = sorted(by_intent)
    pool = {k: list(v) for k, v in by_intent.items()}
    for k in pool:
        rng.shuffle(pool[k])

    order = np.argsort(-sizes, kind="stable")  # largest first, ties by client id
    out: dict[int, list[int]] = {int(c): [] for c in range(len(sizes))}

    # Pass 1 -- Dirichlet draw, capped by availability.
    for cid in order:
        want = int(sizes[cid])
        avail = np.array([len(pool[k]) for k in intents], dtype=float)
        p = rng.dirichlet(np.full(len(intents), alpha)) * (avail > 0)
        if p.sum() == 0:
            continue
        take = np.minimum(rng.multinomial(want, p / p.sum()), avail.astype(int))
        for k, n in zip(intents, take, strict=True):
            if n:
                out[int(cid)].extend(pool[k][:n])
                del pool[k][:n]

    # Pass 2 -- top up anyone short, from whatever remains. Order is by shortfall
    # so the largest gaps are closed first; leftovers are drawn without regard to
    # intent, which is the only neutral way to place them.
    leftover = [i for k in intents for i in pool[k]]
    rng.shuffle(leftover)
    short = sorted(
        ((int(c), int(sizes[c]) - len(out[int(c)])) for c in range(len(sizes))),
        key=lambda t: -t[1],
    )
    for cid, gap in short:
        if gap <= 0:
            continue
        out[cid].extend(leftover[:gap])
        del leftover[:gap]

    if leftover:
        raise AssertionError(f"{len(leftover)} row(s) unplaced after top-up")
    return out


def build(src: pd.DataFrame, alpha: float, seed: int) -> tuple[pd.DataFrame, dict]:
    sizes = src.groupby("household_id").size().sort_index().to_numpy()
    by_intent: dict[str, list[int]] = {str(k): g.index.tolist() for k, g in src.groupby("intent")}

    assigned = assign(sizes, by_intent, alpha, seed)
    rows = []
    for cid, idx in assigned.items():
        sub = src.loc[idx, ["row_index", "worker_id", "intent", "utt"]].copy()
        sub.insert(0, "household_id", cid)
        rows.append(sub)
    part = (
        pd.concat(rows)
        .sort_values(["household_id", "row_index"])
        .reset_index(drop=True)[list(REQUIRED)]
    )

    verify(part, src, sizes)

    per_client = part.groupby("household_id")
    n_utts = per_client.size().to_numpy()
    eff = np.array([effective_intents(g["intent"].value_counts()) for _, g in per_client])
    distinct = per_client["intent"].nunique().to_numpy()
    speakers = per_client["worker_id"].nunique().to_numpy()

    stats = {
        "scheme": "sample_dirichlet_size_matched",
        "source_partition": DEFAULT_SOURCE,
        "alpha": alpha,
        "seed": seed,
        "n_clients": int(len(sizes)),
        "total_utts": int(len(part)),
        "utts_min": int(n_utts.min()),
        "utts_median": float(np.median(n_utts)),
        "utts_max": int(n_utts.max()),
        "intents_median": float(np.median(distinct)),
        "eff_intents_min": round(float(eff.min()), 2),
        "eff_intents_median": round(float(np.median(eff)), 2),
        "eff_intents_max": round(float(eff.max()), 2),
        # The point of the arm: a client is no longer one person. Recording it
        # makes the contrast with the natural partition's `atomic_ok: true`
        # explicit rather than something a reader has to infer.
        "atomic_ok": False,
        "speakers_per_client_median": float(np.median(speakers)),
        "single_intent_clients": int((distinct == 1).sum()),
    }
    return part, stats


def verify(part: pd.DataFrame, src: pd.DataFrame, sizes: np.ndarray) -> None:
    """Every guard here covers a way this could silently produce the wrong arm."""
    if len(part) != len(src):
        raise AssertionError(f"{len(part)} rows placed against {len(src)} in source")
    if sorted(part["row_index"]) != sorted(src["row_index"]):
        raise AssertionError("row set differs from the source -- rows lost or duplicated")
    got = part.groupby("household_id").size().sort_index().to_numpy()
    if not np.array_equal(got, sizes):
        raise AssertionError("client sizes do not match the source; the floors will differ")
    # The failure that would produce two identical arms under different names.
    per_client_speakers = part.groupby("household_id")["worker_id"].nunique()
    if (per_client_speakers <= 1).all():
        raise AssertionError(
            "every client holds one speaker -- speaker atomicity was preserved, so "
            "this is the natural partition rebuilt, not a shard of it."
        )


def sweep(src: pd.DataFrame, args) -> None:
    """Report excess at several alphas. Writes nothing but a scratch parquet.

    The alpha that MATCHES the natural arm's excess is the interesting one, not
    the one that maximises the gap. At matched excess the two arms carry the same
    amount of heterogeneity and differ only in how it is arranged -- so an FL or
    difference is attributable to *structure* (a client is a person vs a
    client is a random bucket) rather than to one arm simply being more skewed.

    Keeping alpha=0.5 as well gives the other half: same structure, different
    magnitude. Together they decompose what a single two-arm comparison confounds.
    """
    from calibrate_divergence import analyse as calibrate

    nat_path = OUT_DIR / "households_identity_min10_calibrated.json"
    target = json.loads(nat_path.read_text())["excess_absolute"] if nat_path.exists() else None

    tmp = OUT_DIR / "_sweep_tmp.parquet"
    print(
        f"\n{'alpha':>7}{'eff intents':>13}{'raw JSD':>10}"
        f"{'floor':>9}{'excess':>9}{'vs natural':>12}"
    )
    best, rows = None, []
    try:
        for a in args.sweep_alpha:
            part, stats = build(src, a, args.seed)
            part.to_parquet(tmp, index=False)
            cal = calibrate(tmp, args.draws, np.random.default_rng(args.null_seed))
            gap = None if target is None else cal["excess_absolute"] - target
            note = "" if gap is None else f"{gap:+.4f}"
            rows.append(
                {
                    "alpha": a,
                    "eff_intents_median": stats["eff_intents_median"],
                    "jsd_observed_mean": cal["jsd_observed_mean"],
                    "jsd_null_mean": cal["jsd_null_mean"],
                    "excess_absolute": cal["excess_absolute"],
                    "ratio": cal["ratio"],
                }
            )
            print(
                f"{a:>7.2f}{stats['eff_intents_median']:>13.2f}"
                f"{cal['jsd_observed_mean']:>10.4f}{cal['jsd_null_mean']:>9.4f}"
                f"{cal['excess_absolute']:>9.4f}{note:>12}"
            )
            if gap is not None and (best is None or abs(gap) < abs(best[1])):
                best = (a, gap)
    finally:
        tmp.unlink(missing_ok=True)

    # Persisted, because the sweep is the evidence for "no alpha matches real
    # owners" and a figure drawn from console output is a figure nothing checks.
    sweep_path = OUT_DIR / "shard_alpha_sweep.json"
    nat_stats = OUT_DIR / "households_identity_min10_stats.json"
    natural = {"excess_absolute": target}
    if nat_stats.exists():
        natural["eff_intents_median"] = json.loads(nat_stats.read_text())["eff_intents_median"]
    sweep_path.write_text(
        json.dumps(
            {
                "source_partition": args.source,
                "draws": args.draws,
                "null_seed": args.null_seed,
                "seed": args.seed,
                "natural": natural,
                "sweep": rows,
            },
            indent=2,
        )
    )
    print(f"\nwrote {sweep_path.relative_to(PROJECT_ROOT)}")

    if target is not None:
        print(f"\n  natural excess {target:.4f}")
        if best:
            print(f"  closest alpha  {best[0]:.2f}  ({best[1]:+.4f} away)")
            print(
                f"\n  Rebuild that arm to keep it:\n"
                f"    python scripts/partition_shard_matched.py --alpha {best[0]} "
                f"--tag shard383_matched --calibrate"
            )
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="shard383")
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="also run the size-matched null calibration, using "
        "calibrate_divergence.analyse (imported, not duplicated)",
    )
    ap.add_argument(
        "--sweep-alpha",
        type=float,
        nargs="+",
        default=None,
        help="try several alphas, report each excess against the natural arm's, "
        "write nothing. Use it to find the alpha whose heterogeneity MATCHES "
        "real speakers -- that arm holds magnitude fixed so a natural-vs-shard "
        "difference is attributable to structure rather than to skew.",
    )
    ap.add_argument("--draws", type=int, default=30, help="null resamples per client")
    ap.add_argument(
        "--null-seed",
        type=int,
        default=42,
        help="Monte Carlo seed for the null ONLY -- must match the source "
        "partition's, or the two floors are not comparable",
    )
    args = ap.parse_args()

    path = Path(args.source)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    src = pd.read_parquet(path)
    missing = sorted(set(REQUIRED) - set(src.columns))
    if missing:
        raise SystemExit(f"{path.name} is missing column(s) {missing}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.sweep_alpha:
        sweep(src, args)
        return

    part, stats = build(src, args.alpha, args.seed)

    pq = OUT_DIR / f"households_{args.tag}.parquet"
    js = OUT_DIR / f"households_{args.tag}_stats.json"
    part.to_parquet(pq, index=False)
    js.write_text(json.dumps(stats, indent=2))

    print(f"\n{pq.relative_to(PROJECT_ROOT)}")
    print(f"  {stats['n_clients']} clients, {stats['total_utts']} rows, alpha={args.alpha}")
    print(
        f"  utts    min {stats['utts_min']}  median {stats['utts_median']:.0f}  "
        f"max {stats['utts_max']}   (matched to source)"
    )
    print(
        f"  eff intents  min {stats['eff_intents_min']}  "
        f"median {stats['eff_intents_median']}  max {stats['eff_intents_max']}"
    )
    print(f"  speakers per client, median {stats['speakers_per_client_median']:.0f}")
    print(f"  single-intent clients: {stats['single_intent_clients']}")
    print("\n  natural arm for comparison: eff intents median 11.42, 1 speaker per client\n")

    if args.calibrate:
        # Imported, not reimplemented -- the null must come from exactly the code
        # that calibrated the natural arm, or the floors are not comparable and
        # the whole size-matching argument collapses.
        from calibrate_divergence import analyse as calibrate

        rng = np.random.default_rng(args.null_seed)
        cal = calibrate(pq, args.draws, rng)
        cal_path = OUT_DIR / f"households_{args.tag}_calibrated.json"
        cal_path.write_text(
            json.dumps(
                {
                    "scheme": "sample_dirichlet_size_matched",
                    "source_partition": args.source,
                    "alpha": args.alpha,
                    "draws": args.draws,
                    "null_seed": args.null_seed,
                    **cal,
                },
                indent=2,
            )
        )

        src_cal = OUT_DIR / "households_identity_min10_calibrated.json"
        print(f"calibration ({args.draws} size-matched resamples per client)")
        print(f"  raw JSD to global  {cal['jsd_observed_mean']:.4f}")
        print(f"  size-matched null  {cal['jsd_null_mean']:.4f}")
        print(f"  excess             {cal['excess_absolute']:.4f}")
        print(f"  ratio              {cal['ratio']:.2f}x")

        if src_cal.exists():
            nat = json.loads(src_cal.read_text())
            # Sizes are copied from the source, so the floor is a property both
            # arms share. A mismatch beyond Monte Carlo noise means the size
            # matching failed and no comparison downstream is interpretable.
            drift = abs(cal["jsd_null_mean"] - nat["jsd_null_mean"])
            print(
                f"\n  natural arm: raw {nat['jsd_observed_mean']:.4f}  "
                f"floor {nat['jsd_null_mean']:.4f}  excess {nat['excess_absolute']:.4f}"
            )
            print(f"  floor agreement:   {drift:.4f} apart", end="")
            print("  OK" if drift < 0.01 else "  <-- SIZES DID NOT MATCH")
            if cal["excess_absolute"] <= nat["excess_absolute"]:
                print(
                    "\n  WARNING: shard excess is not above natural's despite narrower\n"
                    "  clients. Either the shard did not take or the calibration is\n"
                    "  measuring something other than composition."
                )
        print()


if __name__ == "__main__":
    main()
