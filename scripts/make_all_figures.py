"""Regenerate every figure in artifacts/figures/ from the committed JSON.

    python scripts/make_all_figures.py
    python scripts/make_all_figures.py --check     # verify only, write nothing

Why this exists
---------------
Three separate scripts write into ``artifacts/figures/``, and running two of them
leaves the third stale — a figure that silently disagrees with the table beside
it. This is the single entry point, so "regenerate the figures" is one command
and cannot half-happen.

It orchestrates only. No plotting logic lives here; each generator remains
runnable on its own.

Label heatmaps, pairwise-JSD maps and the seed-sweep plot are side effects of
the analysis scripts that compute them; they are not shipped and not covered
here. Those scripts create ``reports/`` on demand:

    python scripts/partition_households.py
    python scripts/analyze_partition_divergence.py --heatmap
    python scripts/sweep_partition_seeds.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from fedknob.data.massive import PROJECT_ROOT

SCRIPTS = Path(__file__).resolve().parent
FIGURES = PROJECT_ROOT / "artifacts" / "figures"

#: (script, what it writes, which JSON it reads) -- kept explicit so a missing
#: input is reported as a data problem rather than a traceback.
GENERATORS = [
    ("make_leverage_figures.py",
     ["fig_alpha_main.png", "fig_alpha_main.pdf",
      "fig_calibration_main.png", "fig_calibration_main.pdf"],
     ["granularity_contrast.json", "calibrated_divergence.json"]),
    ("make_partition_figures.py",
     ["fig_rare_trajectory.png", "fig_rare_trajectory.pdf"],
     ["fedavg_proxy_natural383_r30_s42.json",
      "fedavg_proxy_shardm383_r30_s42.json",
      "fedavg_proxy_shard383_r30_s42.json"]),
]

#: Files a previous version of a generator wrote and nothing produces any more.
#: --check reports them so they do not sit in the folder looking authoritative.
ORPHANS = ["fig_poster_main.png", "fig_poster_main.pdf",
           "fig_poster_alpha.png", "fig_poster_alpha.pdf",
           "fig_poster_calibration.png", "fig_poster_calibration.pdf",
           "fig_calibration.png", "fig_calibration.pdf",
           "fig_alpha_leverage.png", "fig_alpha_granularity.png",
           "fig_seed_stability.png",
           "fig_leverage_both_metrics.png", "fig_leverage_both_metrics.pdf",
           "fig_fedavg_drift.png", "fig_fedavg_drift.pdf",
           "fig_alpha_offcurve.png", "fig_alpha_offcurve.pdf",
           "fig_structure_vs_magnitude.png", "fig_structure_vs_magnitude.pdf"]


#: Inputs are resolved by name across these: partition measurements and
#: federated run histories come from different pipelines and land in different
#: folders, but a generator may read from both.
INPUT_DIRS = (
    PROJECT_ROOT / "artifacts" / "partitions",
    PROJECT_ROOT / "artifacts" / "fl",
)


def missing_inputs(names: list[str]) -> list[str]:
    return [n for n in names if not any((d / n).exists() for d in INPUT_DIRS)]


def main() -> None:
    ap = argparse.ArgumentParser(description="regenerate all figures")
    ap.add_argument("--check", action="store_true",
                    help="report which figures exist and whether their inputs are "
                         "present; write nothing")
    args = ap.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    failures, skipped = [], []

    for script, outputs, inputs in GENERATORS:
        gone = missing_inputs(inputs)
        if gone:
            skipped.append((script, gone))
            print(f"SKIP  {script}")
            for g in gone:
                print(f"        missing {g}")
            continue

        if args.check:
            have = [o for o in outputs if (FIGURES / o).exists()]
            state = "ok" if len(have) == len(outputs) else "INCOMPLETE"
            print(f"{state:<5} {script:<28} {len(have)}/{len(outputs)} present")
            continue

        print(f"---- {script}")
        r = subprocess.run([sys.executable, str(SCRIPTS / script)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            failures.append(script)
            print(r.stdout.strip())
            print(r.stderr.strip(), file=sys.stderr)
        else:
            for line in r.stdout.splitlines():
                if line.startswith("wrote"):
                    print(f"  {line}")

    stale = [o for o in ORPHANS if (FIGURES / o).exists()]
    if stale:
        print("\nORPHANED -- no generator produces these any more; safe to delete:")
        for o in stale:
            print(f"  artifacts/figures/{o}")

    if args.check:
        return

    made = sorted(p.name for p in FIGURES.glob("fig_*"))
    print(f"\n{len(made)} files in artifacts/figures/:")
    for m in made:
        print(f"  {m}")

    if skipped:
        print("\nskipped (missing inputs -- run the analysis script that writes them):")
        for s, g in skipped:
            print(f"  {s}: {', '.join(g)}")
    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
