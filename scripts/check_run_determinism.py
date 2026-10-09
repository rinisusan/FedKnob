"""Does the same command twice give the same run?

    python scripts/check_run_determinism.py
    python scripts/check_run_determinism.py --rounds 5      # faster smoke check

Runs ``run_fedavg.py`` twice with identical arguments into two scratch files and
diffs the trajectories round by round. Exit code 1 if they differ.

Why this exists
---------------
Seeding looked correct long before it was. ``set_seed(cfg["seed"])`` runs inside
``fl/client.py``'s ``get_model``, which executes in the Ray *client workers*. It
fixes everything inside a client's turn -- the model build, the batch order, the
eval split -- and nothing about **which clients get a turn**, because sampling
happens in the server process, which nothing seeded.

The symptom was specific and easy to misread as a code regression: round 0 was
identical to sixteen digits (it precedes any sampling) and every round after it
diverged. Two runs of one command at seed 42 drew ``[0, 14, 50, 62, ...]`` and
``[11, 12, 28, 30, ...]`` in round 1.

``server_fn`` now seeds the server process too. That may or may not be
sufficient -- Flower could sample from a generator that call does not reach, and
Ray's process model is not obliged to make the seeding stick. Reading the code
cannot settle it. Running it twice can, which is what this script does.

Run this after any change to seeding, to the strategy, or to the Flower version.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fedknob.data.massive import PROJECT_ROOT  # noqa: E402

#: Compared round by round. ``sampled`` is first because it is the cause: if the
#: client draws differ, everything downstream must, and reporting a dozen
#: divergent metrics would bury the one that explains them.
KEYS = ["sampled", "accuracy", "f1_macro", "loss", "rare_recall", "target_recall"]


def run(args_extra: list[str], out: Path) -> list[dict]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_fedavg.py"),
        *args_extra,
        "--out",
        str(out),
    ]
    print(f"  $ {' '.join(cmd[1:])}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:], file=sys.stderr)
        hint = ""
        if "CUDA" in r.stderr or "AcceleratorError" in r.stderr:
            hint = (
                "\nThis looks like a CUDA fault rather than a logic error. Ray "
                "force-terminates its actors at shutdown, and starting a second "
                "simulation before the driver has released the first context can "
                "raise cudaErrorIllegalInstruction -- more readily on Blackwell "
                "with the nightly cu128 wheel.\n"
                "  * raise --pause (currently the gap between runs), or\n"
                "  * run the two halves separately:\n"
                "      python scripts/check_run_determinism.py --only a\n"
                "      python scripts/check_run_determinism.py --only b\n"
                "      python scripts/check_run_determinism.py --diff-only"
            )
        raise SystemExit(f"run failed: {out.name}{hint}")
    return json.loads(out.read_text())["history"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--init", default="proxy")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--num-gpus",
        type=float,
        default=None,
        help="fraction of one GPU per client, passed through. 1.0 runs "
        "one client at a time -- slower, but it removes GPU "
        "contention as a variable when a run faults.",
    )
    ap.add_argument(
        "--pause",
        type=float,
        default=25.0,
        help="seconds between the two runs. Ray tears its actors down "
        "asynchronously; starting a second CUDA context too soon "
        "can fault the driver. Raise it if that happens.",
    )
    ap.add_argument(
        "--only",
        choices=["a", "b"],
        help="produce one half and stop -- for running the two by hand "
        "when back-to-back GPU starts are unstable",
    )
    ap.add_argument(
        "--diff-only",
        action="store_true",
        help="compare the two files already on disk, run nothing",
    )
    args = ap.parse_args()

    scratch = PROJECT_ROOT / "artifacts" / "fl" / "_determinism"
    scratch.mkdir(parents=True, exist_ok=True)
    common = [
        "--init",
        args.init,
        "--rounds",
        str(args.rounds),
        "--epochs",
        str(args.epochs),
        "--seed",
        str(args.seed),
    ]
    if args.num_gpus is not None:
        common += ["--num-gpus", str(args.num_gpus)]

    if args.only:
        run(common, scratch / f"{args.only}.json")
        print(f"\nwrote {args.only}.json. Run the other half, then --diff-only.")
        return 0

    if args.diff_only:
        missing = [p.name for p in (scratch / "a.json", scratch / "b.json") if not p.exists()]
        if missing:
            raise SystemExit(f"missing {', '.join(missing)} -- produce them with --only")
        a = json.loads((scratch / "a.json").read_text())["history"]
        b = json.loads((scratch / "b.json").read_text())["history"]
    else:
        print(f"two identical runs, seed {args.seed}, {args.rounds} rounds\n")
        a = run(common, scratch / "a.json")
        if args.pause:
            print(
                f"\n  waiting {args.pause:g}s for Ray and the CUDA context to "
                f"drain before the second run"
            )
            time.sleep(args.pause)
        b = run(common, scratch / "b.json")

    print(f"\n{'round':>6} " + " ".join(f"{k:>14}" for k in KEYS))
    print("-" * (7 + 15 * len(KEYS)))
    diffs: list[str] = []
    for ra, rb in zip(a, b, strict=True):
        marks = []
        for k in KEYS:
            va, vb = ra.get(k), rb.get(k)
            same = va == vb
            if not same:
                diffs.append(f"round {ra['round']} {k}")
            marks.append(("same" if same else "DIFFER").rjust(14))
        print(f"{ra['round']:>6} " + " ".join(marks))

    print()
    if diffs:
        print(f"NOT DETERMINISTIC -- {len(diffs)} divergence(s); first: {diffs[0]}")
        if any(d.endswith("sampled") for d in diffs):
            print("  The client draws differ, so every metric below them must too.")
            print("  Seeding the server process did not reach Flower's sampler.")
        return 1
    print("DETERMINISTIC -- both runs agree on every key, every round.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
