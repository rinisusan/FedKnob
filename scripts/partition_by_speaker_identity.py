"""Fifth partition: one speaker = one client, with a minimum-examples threshold.

    python scripts/partition_by_speaker_identity.py
    python scripts/partition_by_speaker_identity.py --min-utts 20
    python scripts/partition_by_speaker_identity.py --min-utts 0 --tag all

Writes ``artifacts/partitions/households_identity_min{K}.parquet`` plus a stats
JSON, in the SAME schema ``fedknob.data.partition.to_dataframe`` produces
(household_id, row_index, worker_id, intent, utt).

That schema choice is deliberate: **no existing file needs changing.**
``analyze_partition_divergence.py`` already accepts ``--files``, so it reads this
partition unchanged:

    python scripts/analyze_partition_divergence.py \
        --files artifacts/partitions/households_identity_min10.parquet

``calibrate_divergence.py`` resolves paths from client counts and has no
``--files`` option, so rather than edit it we *import* its ``analyse`` function
here and run the calibration in-process, under ``--calibrate``. Identical code,
identical null, no modification to anything already in the repo.

Why this partition exists
-------------------------
The other four partitions apply Dirichlet(alpha) over households and then repair
sizes into a band. That is how benchmarks manufacture non-IID clients. A deployment
cannot do it: a client is a device belonging to one person, its data arrives whole,
and records cannot be reassigned between devices to create skew.

This script builds the partition a deployment actually has. Each speaker becomes
one client. There is no allocation step, so **alpha does not appear** -- not as a
small effect, but as a parameter with nothing to act on. That is the endpoint of
the leverage argument in `sample_level_alpha_sweep.py`, stated structurally rather
than measured.

It is a DIFFERENT SCHEME, not a fifth point on the N=20..200 curve. Those have a
Dirichlet step and enforced size bands; this has neither. Compare the two schemes;
do not fit a trend through both.

Why a threshold rather than a random subset
-------------------------------------------
The raw speaker distribution is severely skewed: median 12 utterances, minimum 1,
and 187 of 682 speakers hold fewer than five. Those clients cannot be split into
train and test in any useful way.

Sampling a random subset does not help -- it preserves the size distribution
exactly while discarding data. Measured on en-US train, a random 400 speakers keeps
61% of the corpus and still leaves 106 clients under five utterances; the 400
largest keep 91% and leave none, because prolific annotators hold most of the data.

A minimum-examples threshold is also what production federated systems do: a device
must hold enough local examples before it is eligible for a round. So this is an
eligibility rule, not a convenience filter, and it should be reported as one.

Determinism
-----------
There is no seed. Grouping by ``worker_id`` involves no randomness, so this
partition has exactly zero seed variance -- against 0.6-4.1% CV for the Dirichlet
partitions. A threshold (rather than "top N") keeps it fully deterministic: no
tie-breaking is needed among speakers with equal counts.

Client ids are assigned by descending utterance count, then by ``worker_id`` as a
stable tie-break, so the mapping is reproducible across runs and platforms.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fedknob.data.massive import PROJECT_ROOT, load_locale  # noqa: E402

OUT_DIR = PROJECT_ROOT / "artifacts" / "partitions"


def effective_intents(counts) -> float:
    """exp(Shannon entropy) -- the same statistic the other partition tools report."""
    c = np.asarray(list(counts), dtype=float)
    if c.sum() <= 0:
        return 0.0
    p = c[c > 0] / c.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def build(df: pd.DataFrame, min_utts: int) -> tuple[pd.DataFrame, dict]:
    """One client per eligible speaker. Returns (partition_frame, stats)."""
    sizes = df.groupby("worker_id").size()
    eligible = sizes[sizes >= min_utts]
    if eligible.empty:
        raise SystemExit(f"no speaker has >= {min_utts} utterances")

    # descending volume, then worker_id: deterministic, no tie-break ambiguity
    order = sorted(eligible.index, key=lambda w: (-int(sizes[w]), str(w)))
    client_of = {w: i for i, w in enumerate(order)}

    keep = df[df["worker_id"].isin(client_of)].copy()
    keep["household_id"] = keep["worker_id"].map(client_of)
    keep["row_index"] = keep.index.astype(int)
    part = (keep[["household_id", "row_index", "worker_id", "intent", "utt"]]
            .sort_values(["household_id", "row_index"])
            .reset_index(drop=True))

    per_client = part.groupby("household_id")
    n_utts = per_client.size().to_numpy()
    eff = np.array([effective_intents(g["intent"].value_counts())
                    for _, g in per_client])
    distinct = per_client["intent"].nunique().to_numpy()

    stats = {
        "scheme": "speaker_identity",
        "min_utts": min_utts,
        "seed": None,                       # deterministic: no randomness anywhere
        "alpha": None,                      # no allocation step for alpha to act on
        "n_clients": int(len(client_of)),
        "n_speakers_total": int(len(sizes)),
        "n_speakers_excluded": int(len(sizes) - len(client_of)),
        "total_utts": int(len(part)),
        "corpus_utts": int(len(df)),
        "corpus_fraction": round(len(part) / len(df), 4),
        "utts_min": int(n_utts.min()),
        "utts_median": float(np.median(n_utts)),
        "utts_max": int(n_utts.max()),
        "clients_under_5_utts": int((n_utts < 5).sum()),
        "clients_under_10_utts": int((n_utts < 10).sum()),
        "intents_median": float(np.median(distinct)),
        "eff_intents_min": round(float(eff.min()), 2),
        "eff_intents_median": round(float(np.median(eff)), 2),
        "eff_intents_max": round(float(eff.max()), 2),
        "atomic_ok": True,                  # true by construction, one speaker per client
    }
    return part, stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description="one speaker = one client, with a minimum-examples threshold")
    ap.add_argument("--locale", default="en-US")
    ap.add_argument("--partition", default="train", choices=("train", "dev", "test", "all"))
    ap.add_argument("--min-utts", type=int, default=10,
                    help="eligibility threshold; 0 keeps every speaker")
    ap.add_argument("--tag", default=None,
                    help="filename suffix (default: min{K})")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--calibrate", action="store_true",
                    help="also run the size-matched null calibration, using "
                         "calibrate_divergence.analyse (imported, not duplicated)")
    ap.add_argument("--draws", type=int, default=30,
                    help="null resamples per client, with --calibrate")
    ap.add_argument("--null-seed", type=int, default=42,
                    help="Monte Carlo seed for the null ONLY. The partition itself has "
                         "no seed; vary this to confirm the null estimate has converged, "
                         "not to measure partition variability.")
    args = ap.parse_args()

    part_arg = None if args.partition == "all" else args.partition
    df = load_locale(args.locale, partition=part_arg).reset_index(drop=True)
    print(f"MASSIVE {args.locale} ({args.partition}): {len(df)} utterances, "
          f"{df['worker_id'].nunique()} speakers, {df['intent'].nunique()} intents")

    frame, stats = build(df, args.min_utts)

    tag = args.tag or f"min{args.min_utts}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq = out_dir / f"households_identity_{tag}.parquet"
    js = out_dir / f"households_identity_{tag}_stats.json"
    frame.to_parquet(pq, index=False)
    js.write_text(json.dumps(stats, indent=2))

    print(f"\nthreshold: >= {args.min_utts} utterances per speaker")
    print(f"  clients            {stats['n_clients']} "
          f"(excluded {stats['n_speakers_excluded']} of {stats['n_speakers_total']})")
    print(f"  utterances kept    {stats['total_utts']} "
          f"({stats['corpus_fraction']:.0%} of corpus)")
    print(f"  utts/client        min {stats['utts_min']}  "
          f"median {stats['utts_median']:.0f}  max {stats['utts_max']}")
    print(f"  under 5 utts       {stats['clients_under_5_utts']}")
    print(f"  eff intents        median {stats['eff_intents_median']}")
    print("  seed               none -- partition is deterministic")
    print(f"\nwrote {pq}")
    print(f"wrote {js}")

    if args.calibrate:
        # Imported, not reimplemented: the null must be computed by exactly the code
        # that produced the numbers for the other four partitions, or the comparison
        # between schemes is not like-for-like.
        from calibrate_divergence import analyse as calibrate

        rng = np.random.default_rng(args.null_seed)
        cal = calibrate(pq, args.draws, rng)
        cal_path = out_dir / f"households_identity_{tag}_calibrated.json"
        cal_path.write_text(json.dumps(
            {"scheme": "speaker_identity", "min_utts": args.min_utts,
             "draws": args.draws, "null_seed": args.null_seed, **cal}, indent=2))

        print(f"\ncalibration ({args.draws} size-matched resamples per client)")
        print(f"  raw JSD to global  {cal['jsd_observed_mean']:.4f}")
        print(f"  size-matched null  {cal['jsd_null_mean']:.4f}")
        print(f"  ratio              {cal['ratio']:.2f}x")
        print(f"  artifact share     {cal['null_share_of_raw']:.0%} of the raw value")
        print("\n  Compare against the Dirichlet partitions in calibrated_divergence.json,")
        print("  but do NOT fit a trend through both: this scheme has no Dirichlet step")
        print("  and no size bands, so granularity is not the only thing that differs.")
        print(f"\nwrote {cal_path}")
    else:
        print("\nnext:")
        print(f"  python scripts/analyze_partition_divergence.py --files {pq}")
        print(f"  python {Path(__file__).name} --min-utts {args.min_utts} --calibrate")


if __name__ == "__main__":
    main()
