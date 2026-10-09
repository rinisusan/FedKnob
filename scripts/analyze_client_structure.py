"""Is client skew *shared* or *idiosyncratic*? The mechanism behind the erosion gap.

    python scripts/analyze_client_structure.py

Writes ``artifacts/partitions/client_structure.json``.

WHY THIS EXISTS
---------------
Natural and size-matched shard partitions carry the same calibrated heterogeneity
(excess 0.1145 vs 0.1259 over an identical 0.3865 floor) and behave completely
differently under FedAvg: rare-intent recall falls 10.16 points on the natural
partition and 0.51 on the shard. The stated explanation is that real owners skew
in *correlated* directions while Dirichlet draws skew independently. This script
measures that claim instead of asserting it.

THE STATISTIC THAT MATTERS: COMMON-MODE FRACTION
------------------------------------------------
For client i with intent distribution ``p_i`` and global distribution ``q``, the
deviation is ``d_i = p_i - q`` -- a direction in 60-dimensional label space saying
what this client over- and under-represents.

FedAvg averages client updates, and a client's update points roughly along its own
``d_i``. So the average keeps what the deviations share and cancels what they do
not. Writing ``dbar`` for the mean deviation, the energy decomposes exactly:

    mean ||d_i||^2  =  ||dbar||^2  +  mean ||d_i - dbar||^2
    (total skew)       (survives)     (cancels)

**common_mode_fraction = ||dbar||^2 / mean ||d_i||^2** is therefore the share of
client skew that averaging cannot remove. It is a number in [0, 1] and it is the
quantity "common-mode drift" was always referring to.

**The prediction was that this would be high for the natural partition and near
zero for the shards. It came out the other way round:**

    natural  0.0011    shard a=1.0  0.0291    shard a=0.5  0.0223

Real owners are the *most* idiosyncratic arm. Their deviations cancel almost
perfectly and they sit closer to global in label space (total energy 0.088 against
0.168 and 0.204) than either shard. Shared label-space drift is therefore **not**
what erodes rare classes, and the paper must not claim it is.

What the shards do have is a systematic pull the real data lacks:
``dirichlet(full(60, alpha))`` is symmetric, so every synthetic client is drawn
toward a *uniform* label distribution while the corpus is skewed -- hence their
5.5x larger ``dbar_l1``. Standard Dirichlet partitioning injects a bias toward
label uniformity that no real population has.

WEIGHTING
---------
FedAvg weights clients by example count, so the size-weighted ``dbar`` is the one
that matches the aggregation. Sizes are identical across the three arms here, so
weighting cannot manufacture a difference between them; both are reported because
the unweighted version is the cleaner statement about the *partition* and the
weighted one about the *algorithm*.

THE SECOND MEASURE: LABEL CO-OCCURRENCE
---------------------------------------
Common-mode fraction says whether skews align. Co-occurrence says why. For each
intent pair, how many clients hold both, against how many would if clients drew
their intents independently. A real speaker who sets alarms also removes them;
a Dirichlet client's intents are an arbitrary subset.

Read the summary statistic with care: natural's ``log2_lift_sd`` (0.365) is *lower*
than the shards' (0.632), which looks like less structure. But the pairs tell a
different story -- natural's strongest are semantically coherent
(``audio_volume_down`` + ``audio_volume_up``, 3.50x) while the shards' are
arbitrary (``music_likeness`` + ``qa_maths``, 4.53x). Random Dirichlet draws over
concentrated clients throw extreme ratios by chance. Natural co-occurrence is
weaker in magnitude but meaningful; shard co-occurrence is stronger but spurious,
and ``log2_lift_sd`` cannot tell the two apart. Do not report it alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedknob.data.massive import PROJECT_ROOT  # noqa: E402

PART = PROJECT_ROOT / "artifacts" / "partitions"
OUT = PART / "client_structure.json"

ARMS = {
    "natural383": "households_identity_min10.parquet",
    "shardm383": "households_shard383_matched.parquet",
    "shard383": "households_shard383.parquet",
}

#: Intent pairs rarer than this under independence are dropped from the lift
#: statistics -- a pair expected in 2 clients that lands in 4 is a 2x lift built
#: from counting noise, and at 60 intents there are 1,770 pairs for noise to
#: exploit.
MIN_EXPECTED_PAIRS = 5.0


def distributions(df: pd.DataFrame, intents: list[str]):
    """Per-client intent distributions ``P`` (N x K), sizes ``n`` (N,)."""
    idx = {k: j for j, k in enumerate(intents)}
    groups = list(df.groupby("household_id", sort=True))
    counts = np.zeros((len(groups), len(intents)))
    for i, (_, g) in enumerate(groups):
        for k, c in g["intent"].value_counts().items():
            counts[i, idx[str(k)]] = c
    n = counts.sum(axis=1)
    return counts / n[:, None], n, counts


def common_mode(P: np.ndarray, q: np.ndarray, w: np.ndarray | None = None) -> dict:
    """Split client deviation energy into the part averaging keeps and drops.

    ``w`` are aggregation weights (FedAvg uses example counts). None = uniform.
    """
    D = P - q  # each row sums to 0
    if w is None:
        w = np.ones(len(D))
    w = w / w.sum()
    dbar = (w[:, None] * D).sum(axis=0)
    total = float((w * (D**2).sum(axis=1)).sum())
    shared = float((dbar**2).sum())
    # cosine between every distinct pair of deviation vectors
    norms = np.linalg.norm(D, axis=1)
    keep = norms > 0
    U = D[keep] / norms[keep, None]
    G = U @ U.T
    iu = np.triu_indices(len(U), k=1)
    return {
        "common_mode_energy": shared,
        "total_energy": total,
        "common_mode_fraction": shared / total if total else 0.0,
        "mean_pairwise_cosine": float(G[iu].mean()),
        "dbar_l1": float(np.abs(dbar).sum()),
    }


def co_occurrence(counts: np.ndarray) -> dict:
    """How far intent co-occurrence within clients departs from independence.

    ``lift = observed / expected``; expected assumes a client's intents are drawn
    independently with the observed per-intent client-coverage marginals. The
    spread of ``log2(lift)`` is the summary: zero means clients look like
    independent draws, large means intents travel in company.
    """
    B = (counts > 0).astype(float)
    n_clients = len(B)
    C = B.T @ B  # pairs of intents held by the same client
    m = B.sum(axis=0)  # clients holding each intent
    E = np.outer(m, m) / n_clients

    iu = np.triu_indices(len(m), k=1)
    obs, exp = C[iu], E[iu]
    keep = exp >= MIN_EXPECTED_PAIRS
    obs, exp = obs[keep], exp[keep]
    # +0.5 keeps log finite where a pair that was expected never appears, which
    # is itself informative (mutually exclusive intents) rather than a nuisance.
    lift = np.log2((obs + 0.5) / (exp + 0.5))
    return {
        "pairs_scored": int(keep.sum()),
        "log2_lift_sd": float(lift.std()),
        "log2_lift_p05": float(np.percentile(lift, 5)),
        "log2_lift_p95": float(np.percentile(lift, 95)),
        "frac_pairs_2x_enriched": float((lift > 1).mean()),
        "frac_pairs_2x_depleted": float((lift < -1).mean()),
        "mean_intents_per_client": float(B.sum(axis=1).mean()),
    }


def top_pairs(counts: np.ndarray, intents: list[str], k: int = 8):
    """The most over-represented intent pairs -- qualitative evidence for the paper."""
    B = (counts > 0).astype(float)
    C = B.T @ B
    m = B.sum(axis=0)
    E = np.outer(m, m) / len(B)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.where(E >= MIN_EXPECTED_PAIRS, (C + 0.5) / (E + 0.5), 0.0)
    iu = np.triu_indices(len(m), k=1)
    order = np.argsort(-L[iu])[:k]
    rows = []
    for o in order:
        a, b = iu[0][o], iu[1][o]
        rows.append({"a": intents[a], "b": intents[b], "lift": round(float(L[a, b]), 2)})
    return rows


def main() -> None:
    frames = {}
    for arm, fn in ARMS.items():
        p = PART / fn
        if not p.exists():
            raise SystemExit(f"missing partition: {p}")
        frames[arm] = pd.read_parquet(p)

    # Every arm must hold the identical row set -- that is what makes this a
    # controlled comparison rather than three unrelated measurements.
    ref = sorted(frames["natural383"]["row_index"])
    for arm, df in frames.items():
        if sorted(df["row_index"]) != ref:
            raise AssertionError(f"{arm} holds a different row set; arms are not matched")

    intents = sorted(frames["natural383"]["intent"].unique())
    gcounts = frames["natural383"]["intent"].value_counts().reindex(intents).to_numpy()
    q = gcounts / gcounts.sum()

    out = {"n_intents": len(intents), "n_rows": len(ref), "arms": {}}
    print(f"rows {len(ref)}  intents {len(intents)}  (identical across arms)\n")
    print(f"{'arm':<12}{'clients':>8}{'common-mode':>13}{'pair cos':>10}{'log2 lift sd':>14}")

    for arm, df in frames.items():
        P, n, counts = distributions(df, intents)
        cm = common_mode(P, q)
        cmw = common_mode(P, q, w=n)
        co = co_occurrence(counts)
        out["arms"][arm] = {
            "n_clients": int(len(P)),
            "unweighted": cm,
            "fedavg_weighted": cmw,
            "co_occurrence": co,
            "top_enriched_pairs": top_pairs(counts, intents),
        }
        print(
            f"{arm:<12}{len(P):>8}{cm['common_mode_fraction']:>13.4f}"
            f"{cm['mean_pairwise_cosine']:>10.4f}{co['log2_lift_sd']:>14.3f}"
        )

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT.relative_to(PROJECT_ROOT)}")

    nat = out["arms"]["natural383"]["unweighted"]["common_mode_fraction"]
    shm = out["arms"]["shardm383"]["unweighted"]["common_mode_fraction"]
    print(
        f"\ncommon-mode fraction: natural {nat:.4f} vs shard-matched {shm:.4f}"
        f"  ({nat / shm:.1f}x)"
        if shm > 0
        else ""
    )
    print(
        "  This is the share of client skew that survives averaging. If it is high\n"
        "  for natural and near zero for the shards, the erosion gap has its\n"
        "  mechanism: real owners pull the same way, Dirichlet clients cancel out."
    )


if __name__ == "__main__":
    main()
