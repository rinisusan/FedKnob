"""Generate the deterministic non-IID household partition + heatmap.

Builds the speaker-grouped, Dirichlet(alpha=0.5) partition of MASSIVE (en) into
200 simulated households, verifies the partition constraints, writes the partition
to artifacts/partitions/households_200.parquet, and renders the label-distribution
heatmap to reports/.

Usage:
    python scripts/partition_households.py                 # full partition
    python scripts/partition_households.py --num-households 50 --quick
    python scripts/partition_households.py --check-byte-identical   # exit-criteria
    python scripts/partition_households.py --alpha-sweep 0.1 0.3 0.5 1.0   # tune skew

Outputs:
    artifacts/partitions/households_200.parquet   -- the partition (committed)
    reports/households_200_label_heatmap.png      -- skew visualisation
    artifacts/partitions/households_200_stats.json-- the verification report
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fedknob.data.massive import PROJECT_ROOT, load_locale
from fedknob.data.partition import (
    DIRICHLET_ALPHA,
    MAX_UTTS,
    MIN_UTTS,
    NUM_HOUSEHOLDS,
    SEED,
    household_label_matrix,
    partition_by_speaker_dirichlet,
    save_partition,
    verify_partition,
)
from fedknob.utils.seeding import set_seed


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def plot_heatmap(matrix: np.ndarray, intent_names: list[str], out_path: Path,
                 alpha: float) -> None:
    """Render the (households x intents) count matrix as a heatmap.

    Rows are reordered by each household's dominant intent so the block-diagonal
    skew is visually obvious -- a uniform (i.i.d.) partition would look flat.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = np.log1p(matrix.astype(float))
    order = np.argsort(matrix.argmax(axis=1), kind="stable")
    shown = shown[order]

    fig, ax = plt.subplots(figsize=(14, 9))
    im = ax.imshow(shown, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("intent")
    ax.set_ylabel("household (sorted by dominant intent)")
    ax.set_title(
        f"MASSIVE (en) non-IID partition - {matrix.shape[0]} households, "
        f"Dirichlet(alpha={alpha})\nlog(1+count); diagonal banding = skew"
    )
    step = max(1, len(intent_names) // 30)
    ax.set_xticks(range(0, len(intent_names), step))
    ax.set_xticklabels([intent_names[i] for i in range(0, len(intent_names), step)],
                       rotation=90, fontsize=6)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log(1 + utterance count)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _print_report(report: dict) -> None:
    print("\n=== partition verification ===")
    for k, v in report.items():
        print(f"  {k:26s}: {v}")
    print(
        "\n  Note: 'intents_*' = raw distinct intents present (inflated because "
        "MASSIVE\n  annotators are generalists). 'eff_intents_*' = exp(entropy) of "
        "the label\n  mix = intents a household *meaningfully* uses -- read the 5-20 "
        "target against\n  THIS, not the raw count."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="non-IID partitioning")
    ap.add_argument("--locale", default="en-US")
    ap.add_argument("--partition", default="train",
                    help="MASSIVE split to partition (train/dev/test/all). "
                         "Default 'train': clients are built from the TRAINING split "
                         "only, leaving dev+test as a pristine global held-out pool "
                         "(for the Week-1/2 baseline yardstick, the Week-5 canary set, "
                         "-- avoids test "
                         "leakage. Use 'all' only if a phase splits train/eval within "
                         "each household.")
    ap.add_argument("--num-households", type=int, default=NUM_HOUSEHOLDS)
    ap.add_argument("--alpha", type=float, default=DIRICHLET_ALPHA)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--min-utts", type=int, default=MIN_UTTS,
                    help="lower bound of the per-household size band. Scale with "
                         "--num-households: at 20 households the mean is ~575, so "
                         "the default [30, 300] is infeasible -- use 300 1200.")
    ap.add_argument("--max-utts", type=int, default=MAX_UTTS,
                    help="upper bound of the per-household size band (see --min-utts)")
    ap.add_argument("--quick", action="store_true",
                    help="smaller household count for a fast smoke run")
    ap.add_argument("--check-byte-identical", action="store_true",
                    help="regenerate twice and assert identical sha256 (exit criterion)")
    ap.add_argument("--alpha-sweep", type=float, nargs="+", default=None,
                    help="report skew metrics across these alphas and exit (no files "
                         "written) -- use to choose alpha against the effective-intent band")
    args = ap.parse_args()

    set_seed(args.seed)
    if args.quick:
        args.num_households = min(args.num_households, 20)

    part_arg = None if args.partition == "all" else args.partition
    df = load_locale(args.locale, partition=part_arg)
    print(f"Loaded MASSIVE {args.locale} ({args.partition}): {len(df)} utterances, "
          f"{df['worker_id'].nunique()} speakers, {df['intent'].nunique()} intents")

    # --- alpha sweep mode: print a table and exit (helps tune the skew) ---
    if args.alpha_sweep:
        print(f"\n{'alpha':>6} | {'distinct intents':>18} | {'effective intents':>20} "
              f"| {'utts':>14}")
        print("-" * 70)
        for a in args.alpha_sweep:
            res = partition_by_speaker_dirichlet(
                df, num_households=args.num_households, alpha=a, seed=args.seed,
                min_utts=args.min_utts, max_utts=args.max_utts,
            )
            r = verify_partition(res, df, min_utts=args.min_utts,
                                 max_utts=args.max_utts)
            print(f"{a:>6} | {r['intents_min']:>4}/{r['intents_median']:>3}/"
                  f"{r['intents_max']:<3}{'':>5} | {r['eff_intents_min']:>5}/"
                  f"{r['eff_intents_median']:<5}/{r['eff_intents_max']:<6} | "
                  f"{r['utts_min']}/{r['utts_median']}/{r['utts_max']}")
        return

    result = partition_by_speaker_dirichlet(
        df, num_households=args.num_households, alpha=args.alpha, seed=args.seed,
        min_utts=args.min_utts, max_utts=args.max_utts,
    )

    report = verify_partition(result, df, min_utts=args.min_utts,
                              max_utts=args.max_utts)
    _print_report(report)

    out_dir = PROJECT_ROOT / "artifacts" / "partitions"
    parquet_path = out_dir / f"households_{args.num_households}.parquet"
    save_partition(result, df, parquet_path)
    print(f"\nPartition  -> {parquet_path}")
    print(f"sha256     -> {_sha256(parquet_path)}")

    (out_dir / f"households_{args.num_households}_stats.json").write_text(
        json.dumps(report, indent=2)
    )

    matrix, names = household_label_matrix(result, df)
    heatmap_path = (PROJECT_ROOT / "reports"
                    / f"households_{args.num_households}_label_heatmap.png")
    plot_heatmap(matrix, names, heatmap_path, args.alpha)
    print(f"Heatmap    -> {heatmap_path}")

    if args.check_byte_identical:
        # MUST pass the same bounds as the first call -- otherwise the regenerated
        # partition uses the module defaults, differs, and reports a spurious FAIL.
        again = partition_by_speaker_dirichlet(
            df, num_households=args.num_households, alpha=args.alpha, seed=args.seed,
            min_utts=args.min_utts, max_utts=args.max_utts,
        )
        tmp = out_dir / f"households_{args.num_households}__verify.parquet"
        save_partition(again, df, tmp)
        same = _sha256(parquet_path) == _sha256(tmp)
        tmp.unlink()
        print(f"\nByte-identical regeneration (seed={args.seed}): "
              f"{'PASS' if same else 'FAIL'}")

    ok = report["coverage_ok"] and report["atomic_ok"]
    print("\nExit criteria:",
          "PASS" if ok else "CHECK",
          "(coverage + speaker-atomicity always required; size/intent bounds "
          "reported above)")


if __name__ == "__main__":
    main()
