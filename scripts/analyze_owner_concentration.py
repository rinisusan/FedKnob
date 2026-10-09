"""Owner-level concentration against a matched-size null (paper §4.1, Appendix K).

    python scripts/analyze_owner_concentration.py
    python scripts/analyze_owner_concentration.py --locales en-US de-DE ja-JP hi-IN
    python scripts/analyze_owner_concentration.py --all --draws 50

Why this script exists
----------------------
§4 originally attributed α's loss of leverage under owner-atomic allocation to owners being
*generalists* -- "a MASSIVE annotator records utterances spanning most of the 60 intents." That
claim is false, and this script is what falsified it. The median en-US annotator contributes 16
utterances and covers 6.5 effective intents, roughly **half** what a random draw of the same size
from the global distribution would cover. Owners are individually concentrated.

What defeats α is not owner breadth but averaging: a client's label distribution is the mean of its
`s` member owners, and owners are narrow in *different* directions, so the mean lands near global
however α assigned them. §4.1 states the corrected mechanism; this script supplies its evidence.

Why the null must be size-matched
---------------------------------
Effective-intent count is bounded above by sample size: an owner with 10 utterances cannot cover
more than 10 intents, so a raw count understates breadth for reasons that have nothing to do with
concentration. Resampling each owner's *actual* size from the pooled distribution puts that ceiling
into the reference value, where it cancels in the ratio. This is the same construction §7 uses at
client level for JSD, applied one level down.

Reads the release archive directly rather than the HuggingFace loader, because the loader exposes
only the locales already registered in ``data/corpus.py`` and the point here is to compare across
locales that are not.
"""
from __future__ import annotations

import argparse
import json
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np

from fedknob.data.massive import PROJECT_ROOT

ARCHIVE = (PROJECT_ROOT / "artifacts" / "massive_raw"
           / "amazon-massive-dataset-1.0.tar.gz")
DEFAULT_LOCALES = ("en-US", "de-DE", "ja-JP", "hi-IN")
OUT = PROJECT_ROOT / "artifacts" / "partitions" / "owner_concentration.json"


def effective_classes(counts) -> float:
    """exp(Shannon entropy) over a count vector -- the same statistic §4 reports."""
    c = np.asarray(list(counts), dtype=float)
    if c.sum() <= 0:
        return 0.0
    p = c / c.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def available_locales(archive: Path) -> list[str]:
    with tarfile.open(archive, "r:gz") as tf:
        return sorted(
            Path(m.name).stem for m in tf.getmembers()
            if m.name.endswith(".jsonl") and "/data/" in m.name
        )


def load_locale_rows(archive: Path, locale: str) -> list[dict]:
    """Stream one locale's jsonl straight out of the tarball -- no extraction to disk."""
    member = f"1.0/data/{locale}.jsonl"
    with tarfile.open(archive, "r:gz") as tf:
        try:
            fh = tf.extractfile(member)
        except KeyError:
            raise SystemExit(f"{member} not found in {archive.name}") from None
        if fh is None:
            raise SystemExit(f"could not read {member}")
        return [json.loads(line) for line in fh.read().decode("utf-8").splitlines() if line]


def analyse(rows: list[dict], draws: int, rng: np.random.Generator) -> dict:
    """Observed vs matched-size-null effective intents, per owner."""
    global_counts = Counter(r["intent"] for r in rows)
    labels = sorted(global_counts)
    gp = np.array([global_counts[k] for k in labels], dtype=float)
    gp /= gp.sum()

    by_worker: dict[str, list[str]] = {}
    for r in rows:
        by_worker.setdefault(r["worker_id"], []).append(r["intent"])

    observed, null, sizes = [], [], []
    for ints in by_worker.values():
        n = len(ints)
        sizes.append(n)
        observed.append(effective_classes(Counter(ints).values()))
        # matched size: n i.i.d. draws from the pooled label distribution
        null.append(float(np.mean([
            effective_classes(Counter(rng.choice(len(labels), size=n, p=gp).tolist()).values())
            for _ in range(draws)
        ])))

    observed = np.array(observed)
    null = np.array(null)
    sizes = np.array(sizes, dtype=float)
    med_obs, med_null = float(np.median(observed)), float(np.median(null))
    return {
        "n_rows": len(rows),
        "n_workers": len(by_worker),
        "median_utts_per_worker": float(np.median(sizes)),
        "min_utts_per_worker": int(sizes.min()),
        "max_utts_per_worker": int(sizes.max()),
        "n_intents": len(labels),
        "median_observed_effective": round(med_obs, 4),
        "median_null_effective": round(med_null, 4),
        "ratio_observed_over_null": round(med_obs / med_null, 4) if med_null else None,
        "median_of_per_worker_ratios": round(float(np.median(observed / null)), 4),
        "max_feasible_owner_atomic_N": len(by_worker),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="owner-level concentration vs a matched-size null")
    ap.add_argument("--archive", default=str(ARCHIVE))
    ap.add_argument("--locales", nargs="*", default=list(DEFAULT_LOCALES))
    ap.add_argument("--all", action="store_true", help="every locale in the archive (slow)")
    ap.add_argument("--draws", type=int, default=20,
                    help="resamples per owner for the null; raise for tighter estimates on "
                         "locales with few utterances per worker")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    archive = Path(args.archive)
    if not archive.exists():
        raise SystemExit(f"archive not found: {archive}")

    locales = available_locales(archive) if args.all else args.locales
    rng = np.random.default_rng(args.seed)

    print(f"archive: {archive.name}")
    print(f"null: {args.draws} size-matched resamples per owner, seed {args.seed}\n")
    hdr = (f"{'locale':<8} {'workers':>7} {'utts/wkr':>9} {'observed':>9} {'null':>8} "
           f"{'ratio':>7} {'max N':>6}")
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for loc in locales:
        r = analyse(load_locale_rows(archive, loc), args.draws, rng)
        results[loc] = r
        print(f"{loc:<8} {r['n_workers']:>7d} {r['median_utts_per_worker']:>9.0f} "
              f"{r['median_observed_effective']:>9.2f} {r['median_null_effective']:>8.2f} "
              f"{r['ratio_observed_over_null']:>7.3f} {r['max_feasible_owner_atomic_N']:>6d}")

    print("\nratio ~1.0  -> owners indistinguishable from random draws (generalists)")
    print("ratio <<1.0 -> owners concentrated on a narrow set of intents (specialists)")

    if "en-US" in results and len(results) > 1:
        en = results["en-US"]["n_workers"]
        others = {k: v["n_workers"] for k, v in results.items() if k != "en-US"}
        if others and max(others.values()) < en / 4:
            print("\nNOTE: en-US utterances were AUTHORED by crowdworkers; the other locales are")
            print("translations of the same 16,521 utterances, so worker_id there identifies a")
            print("translator rather than an independent author. Owner-atomic partitioning is")
            print(f"capped at max N = {max(others.values())} outside en-US (Appendix K.2).")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "locales": results}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
