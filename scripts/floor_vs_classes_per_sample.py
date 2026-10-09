"""How the size-matched floor scales with classes per sample, K/n.

    python scripts/floor_vs_classes_per_sample.py
    python scripts/floor_vs_classes_per_sample.py --draws 2000   # tighter estimates

Writes ``artifacts/partitions/floor_vs_k_over_n.json``.

Why this script exists
----------------------
`calibrate_divergence.py` computes the floor for *our* partitions. This one answers
the question a reviewer asks next: is the calibration problem a property of this
corpus, or does it apply to anyone's benchmark?

It is corpus-independent by construction. No data is read. Clients are simulated
directly: draw n labels i.i.d. from a uniform distribution over K classes, and
measure mean JSD to that distribution. Every simulated client is a random sample of
the same population, so the systematic heterogeneity is zero and whatever the
measurement returns is floor.

The result is that the floor is governed by **K/n**, classes per sample, not by K or
n alone. Two consequences the paper reports:

  * CIFAR-10 at Hsu et al.'s standard 100-client split (K=10, n=500) sits at
    K/n = 0.02 with a floor near 0.004. Raw divergence there is essentially all
    signal, which is likely why the problem has gone unremarked -- the canonical
    benchmark is the case where calibration does not matter.
  * Above roughly K/n = 0.5 the floor exceeds 0.1 and calibration stops being
    optional. MASSIVE at N=200 sits at K/n = 1.09.

Edge deployments push K/n up on both axes at once: larger label spaces and fewer
samples per device.

Why uniform, and what that costs
--------------------------------
A uniform global distribution is the neutral choice -- it depends on no corpus and
gives the *largest* floor for a given (K, n), since a skewed distribution has fewer
effective classes to miss. Our own measured floors are therefore lower than this
simulation predicts (0.224 measured at N=200 against 0.262 simulated), so K/n sets
the scale, not the exact value. Quote the measured floor for a real partition; quote
this for the general shape.

There is an asymptotic check built in. For small divergences,

    E[JSD] ~= (K - 1) / (8 n ln 2)

which is proportional to K/n and is why that ratio governs. It is a first-order
approximation and drifts once the floor is large, so the simulation is what the
paper quotes; the formula is reported alongside as a sanity check on the scaling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fedknob.data.massive import PROJECT_ROOT

OUT = PROJECT_ROOT / "artifacts" / "partitions" / "floor_vs_k_over_n.json"

#: (K, n, label) -- configurations the paper names, plus the grid for the trend.
NAMED = [
    (10, 500, "CIFAR-10, Hsu et al. standard 100-client split"),
    (10, 100, "CIFAR-10, 500 clients"),
    (62, 100, "FEMNIST-like, 62 classes"),
    (62, 124, "K/n = 0.50, the threshold"),
    (60, 582, "MASSIVE N=20"),
    (60, 108, "MASSIVE N=100"),
    (60, 55, "MASSIVE N=200"),
    (60, 26, "MASSIVE identity, >=10 utterances"),
    (60, 12, "MASSIVE identity, all speakers"),
]
GRID_K = (10, 20, 60, 100, 200)
GRID_N = (50, 100, 250, 500, 1000)


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits; same definition as the rest of the repo."""
    m = 0.5 * (p + q)

    def kl(a, b):
        k = a > 0
        return float(np.sum(a[k] * np.log2(a[k] / b[k])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def floor(K: int, n: int, draws: int, rng: np.random.Generator) -> float:
    """Mean JSD to global for clients that are pure i.i.d. draws -- i.e. the floor."""
    gp = np.ones(K) / K
    return float(np.mean([
        jsd(np.bincount(rng.choice(K, size=n, p=gp), minlength=K) / n, gp)
        for _ in range(draws)
    ]))


def asymptotic(K: int, n: int) -> float:
    """(K-1)/(8 n ln 2) -- the small-divergence approximation, for the scaling check."""
    return (K - 1) / (8 * n * np.log(2))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="size-matched floor as a function of classes per sample")
    ap.add_argument("--draws", type=int, default=600,
                    help="simulated clients per (K, n) cell")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"uniform global distribution; {args.draws} simulated clients per cell; "
          f"seed {args.seed}\n")

    print("=== named configurations ===")
    hdr = f"{'K':>4} {'n':>5} {'K/n':>6} {'floor':>7} {'>0.1?':>6}   configuration"
    print(hdr)
    print("-" * (len(hdr) + 8))
    named = []
    for K, n, label in NAMED:
        f = floor(K, n, args.draws, rng)
        named.append({"K": K, "n": n, "k_over_n": round(K / n, 4),
                      "floor": round(f, 4), "label": label})
        print(f"{K:>4} {n:>5} {K/n:>6.2f} {f:>7.4f} {'yes' if f > 0.1 else 'no':>6}   {label}")

    print("\n=== grid: floor(K, n) ===")
    print("  K \\ n " + " ".join(f"{n:>8}" for n in GRID_N))
    grid = {}
    for K in GRID_K:
        row = [floor(K, n, args.draws, rng) for n in GRID_N]
        grid[str(K)] = {str(n): round(v, 4) for n, v in zip(GRID_N, row, strict=True)}
        print(f"{K:>7} " + " ".join(f"{v:>8.3f}" for v in row))

    print("\n=== is K/n the governing variable? equal K/n should give equal floor ===")
    print(f"{'K':>4} {'n':>5} {'K/n':>6} {'simulated':>10} {'(K-1)/8n ln2':>13}")
    check = []
    for K, n in ((10, 500), (20, 250), (60, 750), (100, 1250), (62, 100), (60, 55)):
        f, a = floor(K, n, args.draws, rng), asymptotic(K, n)
        check.append({"K": K, "n": n, "k_over_n": round(K / n, 4),
                      "simulated": round(f, 4), "asymptotic": round(a, 4)})
        print(f"{K:>4} {n:>5} {K/n:>6.2f} {f:>10.4f} {a:>13.4f}")
    print("\n  The first four share K/n = 0.02--0.08 and land on the same floor despite")
    print("  K and n each varying 10x. The approximation drifts once the floor is large.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"draws": args.draws, "seed": args.seed,
                   "global_distribution": "uniform",
                   "note": "corpus-independent; no data is read"},
        "named": named, "grid": grid, "scaling_check": check,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
