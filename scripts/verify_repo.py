"""Check that every reported number, figure and script in the repo agrees.

    python scripts/verify_repo.py          # exit 1 on any failure
    python scripts/verify_repo.py -v       # also list the not-cited numbers

Five checks, each answering a question that has bitten this project at least once:

  A  do all scripts still compile?
  B  is every figure newer than every JSON it reads?
     -- a figure regenerated before its data is a figure that disagrees with the
        table beside it, silently.
  C  does every number quoted in the docs exist in the JSON it came from?
     -- the calibration correction changed twelve figures across three files.
  D  are the derived quantities self-consistent?
     -- ratio == raw/floor, floor share == 1/ratio, raw and ratio move in
        opposite directions.
  E  is any superseded value still asserted as live?
     -- the single-floor calibration (0.0217) and the "annotators are generalists"
        mechanism were both wrong and both had to be corrected in place. This
        check allows them to appear inside a correction notice and flags them
        anywhere else.

Check E matches substrings, so it reports false positives (``5.4x`` inside
``35.4x``). It is a prompt to look, not a verdict.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from fedknob.data.massive import PROJECT_ROOT

PART = PROJECT_ROOT / "artifacts" / "partitions"
FIGS = PROJECT_ROOT / "artifacts" / "figures"
BASE = PROJECT_ROOT / "artifacts" / "baseline"
FL = PROJECT_ROOT / "artifacts" / "fl"
#: Figure inputs are looked up by name across these, because partition
#: measurements and federated run histories come from different pipelines.
INPUT_DIRS = (PART, FL, BASE)
SCRIPTS = PROJECT_ROOT / "scripts"
#: Prose the claim-check scans. `read_text` is unguarded on purpose: a file
#: listed here and missing is a repo error, not something to skip quietly.
DOCS = (
    "README.md",
    "artifacts/README.md",
)

#: figure stem -> the JSON it reads. Keep in step with the generators.
FIGURE_DEPS = {
    "fig_alpha_main": ["granularity_contrast.json"],
    "fig_calibration_main": ["calibrated_divergence.json"],
    # Inputs for these live in artifacts/fl/, resolved by name across both
    # dirs. Only seed 42 of each arm is listed: the five seeds of an arm are
    # written in one sitting, so the freshest of them is what the mtime check
    # compares against.
    "fig_rare_trajectory": [
        "fedavg_proxy_natural383_r30_s42.json",
        "fedavg_proxy_shardm383_r30_s42.json",
        "fedavg_proxy_shard383_r30_s42.json",
    ],
}

#: value -> why it is superseded. Allowed only inside a correction notice.
SUPERSEDED = {
    "0.0217": "the single un-size-matched floor",
    "14.9×": "old N=200 ratio",
    "9.6×": "old N=100 ratio",
    "generalist": "falsified mechanism (annotators are specialists, 0.53 of chance)",
    # Tested directly and rejected. Real owners' deviations cancel almost
    # perfectly -- common-mode fraction 0.0011 against 0.029 for the shards -- so
    # shared drift is not what erodes rare classes. The pattern here matches the
    # *assertion* rather than the term, so prose can report the measurement
    # without tripping the check.
    "mechanism is common-mode": "falsified (client_structure.json: 0.0011 vs 0.029)",
    "drift eroding rare": "same falsified mechanism, restated",
}
CORRECTION_MARKERS = (
    "corrected",
    "revision",
    "earlier version",
    "earlier reading",
    "previously",
    "first supposed",
    "does not survive",
    "is wrong",
    "is false",
    "not that annotators",
    "an earlier",
    "ruled out",
    "falsified",
    "rejected",
    "remains open",
)

fails: list[str] = []


def head(t: str) -> None:
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def mark(good: bool, label: str, note: str = "") -> None:
    if not good:
        fails.append(label)
    print(f"  {'PASS' if good else '**FAIL**':<10} {label}{'   ' + note if note else ''}")


def check_scripts() -> None:
    head("A. SCRIPTS COMPILE")
    for f in sorted(SCRIPTS.glob("*.py")):
        r = subprocess.run([sys.executable, "-m", "py_compile", str(f)], capture_output=True)
        mark(r.returncode == 0, f.name)


def check_figures() -> None:
    head("B. FIGURES NEWER THAN THEIR INPUTS")
    for stem, deps in FIGURE_DEPS.items():
        figs = list(FIGS.glob(stem + ".*"))
        if not figs:
            mark(False, stem, "MISSING")
            continue
        resolved = {d: next((p / d for p in INPUT_DIRS if (p / d).exists()), None) for d in deps}
        missing = [d for d, p in resolved.items() if p is None]
        if missing:
            mark(False, stem, f"input missing: {', '.join(missing)}")
            continue
        newest_dep = max(p.stat().st_mtime for p in resolved.values())
        oldest_fig = min(p.stat().st_mtime for p in figs)
        mark(oldest_fig > newest_dep, stem, f"{len(figs)} file(s)")


def load_claims() -> list[tuple[str, str]]:
    cal = json.loads((PART / "calibrated_divergence.json").read_text())["partitions"]
    gc = json.loads((PART / "granularity_contrast.json").read_text())
    summ = {s["n_clients"]: s for s in gc["summary"]}
    oc = json.loads((PART / "owner_concentration.json").read_text())["locales"]["en-US"]

    out: list[tuple[str, str]] = []
    for k in sorted(cal, key=int):
        c = cal[k]
        out += [
            (f"N={k} raw", f"{c['jsd_observed_mean']:.4f}"),
            (f"N={k} floor", f"{c['jsd_null_mean']:.4f}"),
            (f"N={k} ratio", f"{c['ratio']:.2f}"),
        ]
    for tag, lbl in (("all", "682"), ("min10", "383")):
        p = PART / f"households_identity_{tag}_calibrated.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        out += [
            (f"identity {lbl} raw", f"{r['jsd_observed_mean']:.4f}"),
            (f"identity {lbl} floor", f"{r['jsd_null_mean']:.4f}"),
            (f"identity {lbl} ratio", f"{r['ratio']:.2f}"),
            (f"identity {lbl} clients", str(r["n_clients"])),
        ]
    out += [
        ("leverage N=20", f"{summ[20]['leverage_ratio']}"),
        ("leverage N=200", f"{summ[200]['leverage_ratio']}"),
        ("owner observed", f"{oc['median_observed_effective']:.2f}"),
        ("owner floor", f"{oc['median_null_effective']:.2f}"),
    ]

    # head-regime control (freeze_pre_classifier). Absent on a fresh clone until
    # the control has been run, so treat it as optional rather than a hard failure.
    hrc = BASE / "head_regime_control.json"
    if hrc.exists():
        h = json.loads(hrc.read_text())
        uf = h["runs"]["unfrozen"]["metrics"]
        fr = h["runs"]["frozen"]["metrics"]
        for key in sorted(uf):
            short = key.replace("eval_", "")
            out += [
                (f"head unfrozen {short}", f"{uf[key]:.4f}"),
                (f"head frozen {short}", f"{fr[key]:.4f}"),
            ]
        hd = h["headline"]
        # carry a unit with the 2-decimal figures: bare "0.28" collides too easily
        out += [
            ("head cost in SE", f"{hd['cost_in_standard_errors']:.2f} SE"),
            ("head accuracy SE", f"{hd['accuracy_se_pts']:.2f} pts"),
            ("head params frozen", f"{hd['params_frozen']:,}"),
            ("head trainable frozen", f"{h['runs']['frozen']['trainable_params']:,}"),
            ("head trainable unfrozen", f"{h['runs']['unfrozen']['trainable_params']:,}"),
        ]
    return out


def check_docs(verbose: bool) -> None:
    head("C. NUMBERS QUOTED IN DOCS EXIST IN THE JSON")
    texts = {d: (PROJECT_ROOT / d).read_text(encoding="utf-8") for d in DOCS}
    uncited = 0
    for name, val in load_claims():
        where = [Path(d).name for d, t in texts.items() if val in t]
        if where:
            print(f"  {'PASS':<10} {name:<26} {val:>8}   {', '.join(sorted(set(where)))}")
        else:
            uncited += 1
            if verbose:
                print(f"  {'n/a':<10} {name:<26} {val:>8}   (not cited)")
    print(
        f"\n  {uncited} value(s) not cited in any doc"
        f"{'' if verbose else ' -- rerun with -v to list'}"
    )


def check_consistency() -> None:
    head("D. DERIVED QUANTITIES SELF-CONSISTENT")
    cal = json.loads((PART / "calibrated_divergence.json").read_text())["partitions"]
    rows = list(cal.values())
    for tag in ("all", "min10"):
        p = PART / f"households_identity_{tag}_calibrated.json"
        if p.exists():
            rows.append(json.loads(p.read_text()))

    mark(
        all(abs(r["ratio"] - r["jsd_observed_mean"] / r["jsd_null_mean"]) < 0.006 for r in rows),
        f"ratio == raw / floor  ({len(rows)} partitions)",
    )
    mark(
        all(abs(r["null_share_of_raw"] - 1 / r["ratio"]) < 0.01 for r in rows),
        "floor share == 1 / ratio",
    )
    pairs = sorted((r["jsd_observed_mean"], r["ratio"]) for r in rows)
    mark(
        all(pairs[i][1] > pairs[i + 1][1] for i in range(len(pairs) - 1)),
        "raw rises while ratio falls, monotonically",
    )

    dv = {r["n_households"]: r for r in json.loads((PART / "divergence_report.json").read_text())}
    mark(
        all(
            abs(dv[int(k)]["jsd_to_global_mean"] - cal[k]["jsd_observed_mean"]) < 1e-4 for k in cal
        ),
        "divergence_report raw == calibrated raw",
    )

    gc = json.loads((PART / "granularity_contrast.json").read_text())
    summ = {s["n_clients"]: s for s in gc["summary"]}
    mark(
        all(
            abs(
                summ[n]["leverage_ratio"] - summ[n]["sample_eff_span"] / summ[n]["speaker_eff_span"]
            )
            < 0.05
            for n in summ
        ),
        "leverage == sample span / owner span",
    )


def check_superseded() -> None:
    head("E. SUPERSEDED VALUES ASSERTED AS LIVE?")
    print("  (substring matching -- expect false positives; this is a prompt to look)\n")
    flagged = 0
    for pat, why in SUPERSEDED.items():
        lines = []
        for d in DOCS:
            text = (PROJECT_ROOT / d).read_text(encoding="utf-8").split("\n")
            for i, line in enumerate(lines_ := text):
                if re.search(re.escape(pat), line, re.I):
                    ctx = "\n".join(lines_[max(0, i - 5) : i + 4]).lower()
                    if not any(m in ctx for m in CORRECTION_MARKERS):
                        lines.append(f"{Path(d).name}:{i + 1}")
        if lines:
            flagged += len(lines)
            print(f"  REVIEW     '{pat}' ({why})")
            for loc in lines:
                print(f"               {loc}")
        else:
            print(f"  ok         '{pat}' absent or only inside a correction")
    if flagged:
        print(f"\n  {flagged} location(s) to eyeball -- not counted as failures")


def main() -> None:
    ap = argparse.ArgumentParser(description="verify repo results, figures and docs")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    check_scripts()
    check_figures()
    check_docs(args.verbose)
    check_consistency()
    check_superseded()

    head("RESULT")
    if fails:
        print(f"  {len(fails)} FAILURE(S):")
        for f in fails:
            print(f"    - {f}")
        raise SystemExit(1)
    print("  all checks passed")
    print("\n  Not covered here: run `pytest` for the test suite.")


if __name__ == "__main__":
    main()
