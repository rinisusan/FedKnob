"""Week 3 -- Non-IID partitioning of MASSIVE by speaker (worker_id).

Phase I, Week 3 goal (from the implementation plan):
    * Group MASSIVE utterances by speaker_id; assign 200 simulated households.
    * Apply Dirichlet(alpha=0.5) over intent labels to skew the label distribution.
    * Verify each household has 30-300 utterances and 5-20 intents represented.
    * Save a deterministic partition to artifacts/partitions/households_200.parquet.
    * Exit criteria: the partition file regenerates byte-identical from seed=42.

Why this module exists
----------------------
The federated phases (Week 4 onward) need a *fixed* map from utterance -> household
that is (a) realistically non-IID -- households talk about different things -- and
(b) byte-for-byte reproducible so every experiment reads identical clients. This
module is the single source of truth for that map.

Design: speaker-level Dirichlet label skew
------------------------------------------
A MASSIVE ``worker_id`` (the annotator who recorded an utterance) is the atomic
unit -- a worker's utterances are NEVER split across households. This is the
"household" proxy: one speaker, one device. Keeping a speaker whole is what makes
the heterogeneity *realistic* rather than synthetic i.i.d. noise.

On top of speaker grouping we layer the classic Dirichlet label-skew of
Hsu, Qi & Brown (2019, arXiv:1909.06335), but applied at *speaker* granularity:

    For each intent c:
        1. collect the speakers whose dominant intent is c (sorted by worker_id);
        2. draw proportions  p_c ~ Dirichlet(alpha)  of length = num_households;
        3. split that intent's speakers across the households following p_c.

A small ``alpha`` (we use 0.5) makes each draw lopsided, so most of an intent's
speakers land in a few households -> strong, controllable non-IID skew. As
``alpha -> infinity`` the split becomes uniform (i.i.d.); ``alpha -> 0`` makes
each intent collapse onto a single household. 0.5 is the standard "moderately
heterogeneous" operating point used throughout the FL literature.

On the "5-20 intents represented" target
----------------------------------------
MASSIVE ``worker_id``s are crowd annotators, not real end users: each annotator
recorded utterances spanning most of the 60 intents. So a single atomic speaker
already carries ~20+ distinct intents, and any household built from whole
speakers necessarily *represents* many intents (empirically 15-43 on en-US),
independent of ``alpha``. The realistic non-IID signal therefore lives in the
label *proportions* (which Dirichlet skews strongly), not in the *support*. To
measure the proportion skew honestly we report, alongside the raw distinct-intent
count, an **effective intent count** = exp(entropy of the per-household label
distribution): the number of intents a household *meaningfully* uses. That is the
quantity the plan's "5-20 intents" target was really after.

After assignment a deterministic repair pass moves whole speakers from the
largest households into the smallest ones until every household satisfies the
size bounds (best-effort; reported by :func:`verify_partition`).

Public API
----------
    partition_by_speaker_dirichlet(df, ...)  -> PartitionResult
    verify_partition(result, df, ...)        -> dict (constraint report)
    save_partition(result, df, path)         -> writes the parquet
    load_partition(path)                     -> reads it back
    household_label_matrix(result, df, ...)  -> (households x intents) count matrix
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Week 3 defaults -- the proposal's numbers, pinned here as the single source.
NUM_HOUSEHOLDS = 200
DIRICHLET_ALPHA = 0.5
SEED = 42
MIN_UTTS, MAX_UTTS = 30, 300
MIN_INTENTS, MAX_INTENTS = 5, 20


@dataclass
class PartitionResult:
    """The output of a partitioning run."""

    assignment: dict[int, list[int]]
    speaker_to_household: dict[object, int]
    num_households: int
    alpha: float
    seed: int


# ----------------------------------------------------------------------------- #
# Core partitioning
# ----------------------------------------------------------------------------- #
def _speaker_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per speaker: worker_id, dominant intent, #utts, row indices."""
    records = []
    for worker_id, idx in sorted(
        df.groupby("worker_id").indices.items(), key=lambda kv: str(kv[0])
    ):
        rows = sorted(int(i) for i in idx)
        intents = df.iloc[rows]["intent"]
        counts = intents.value_counts()
        top = counts.max()
        dominant = sorted(counts[counts == top].index)[0]
        records.append(
            {"worker_id": worker_id, "dominant_intent": dominant, "n_utts": len(rows), "rows": rows}
        )
    return pd.DataFrame.from_records(records)


def _dirichlet_split_indices(n_items: int, proportions: np.ndarray) -> np.ndarray:
    """Map ``n_items`` ordered items to bins given target ``proportions``."""
    if n_items == 0:
        return np.empty(0, dtype=int)
    p = np.asarray(proportions, dtype=float)
    p = p / p.sum()
    raw = p * n_items
    sizes = np.floor(raw).astype(int)
    remainder = n_items - sizes.sum()
    if remainder > 0:
        order = np.argsort(-(raw - sizes), kind="stable")
        sizes[order[:remainder]] += 1
    return np.repeat(np.arange(len(p)), sizes)


def partition_by_speaker_dirichlet(
    df: pd.DataFrame,
    num_households: int = NUM_HOUSEHOLDS,
    alpha: float = DIRICHLET_ALPHA,
    seed: int = SEED,
    min_utts: int = MIN_UTTS,
    max_utts: int = MAX_UTTS,
    repair_sizes: bool = True,
) -> PartitionResult:
    """Partition ``df`` into ``num_households`` non-IID households."""
    rng = np.random.default_rng(seed)
    speakers = _speaker_table(df)
    if len(speakers) < num_households:
        raise ValueError(
            f"only {len(speakers)} distinct speakers but num_households="
            f"{num_households}; reduce num_households (each household needs "
            f">=1 speaker, and speakers are atomic)."
        )

    assignment: dict[int, list[int]] = {h: [] for h in range(num_households)}
    speaker_to_household: dict[object, int] = {}

    for intent in sorted(speakers["dominant_intent"].unique()):
        grp = speakers[speakers["dominant_intent"] == intent]
        grp = grp.sort_values("worker_id", key=lambda s: s.astype(str))
        proportions = rng.dirichlet(np.full(num_households, alpha))
        bins = _dirichlet_split_indices(len(grp), proportions)
        perm = rng.permutation(num_households)
        for (_, spk), b in zip(grp.iterrows(), bins, strict=True):
            h = int(perm[b])
            assignment[h].extend(spk["rows"])
            speaker_to_household[spk["worker_id"]] = h

    if repair_sizes:
        assignment, speaker_to_household = _repair_sizes(
            df, assignment, speaker_to_household, min_utts, max_utts, rng
        )

    assignment = {h: sorted(rows) for h, rows in assignment.items()}
    return PartitionResult(
        assignment=assignment,
        speaker_to_household=speaker_to_household,
        num_households=num_households,
        alpha=alpha,
        seed=seed,
    )


def _household_sizes(assignment: dict[int, list[int]]) -> dict[int, int]:
    return {h: len(rows) for h, rows in assignment.items()}


def _repair_sizes(
    df: pd.DataFrame,
    assignment: dict[int, list[int]],
    speaker_to_household: dict[object, int],
    min_utts: int,
    max_utts: int,
    rng: np.random.Generator,
    max_passes: int = 10_000,
) -> tuple[dict[int, list[int]], dict[object, int]]:
    """Move whole speakers from oversized to undersized households (feasibility-aware)."""
    rows_by_speaker: dict[object, list[int]] = {}
    for _h, rows in assignment.items():
        for r in rows:
            w = df.iloc[r]["worker_id"]
            rows_by_speaker.setdefault(w, []).append(r)

    def speakers_in(h: int) -> list[object]:
        spk = [w for w, hh in speaker_to_household.items() if hh == h]
        return sorted(spk, key=lambda w: (len(rows_by_speaker[w]), str(w)))

    def move(spk: object, src: int, dst: int) -> None:
        moved = set(rows_by_speaker[spk])
        assignment[src] = [r for r in assignment[src] if r not in moved]
        assignment[dst].extend(rows_by_speaker[spk])
        speaker_to_household[spk] = dst

    for _ in range(max_passes):
        sizes = _household_sizes(assignment)
        under = sorted((h for h in sizes if sizes[h] < min_utts), key=lambda h: (sizes[h], h))
        over = sorted((h for h in sizes if sizes[h] > max_utts), key=lambda h: (-sizes[h], h))
        if not under and not over:
            break
        progressed = False

        for hi in over:
            for spk in speakers_in(hi):
                s = len(rows_by_speaker[spk])
                if sizes[hi] - s < min_utts and len(speakers_in(hi)) > 1:
                    continue
                cands = sorted(
                    (h for h in sizes if h != hi and sizes[h] + s <= max_utts),
                    key=lambda h: (sizes[h], h),
                )
                if cands:
                    move(spk, hi, cands[0])
                    progressed = True
                    break
            if progressed:
                break
        if progressed:
            continue

        for lo in under:
            for donor in sorted((h for h in sizes if h != lo), key=lambda h: (-sizes[h], h)):
                if len(speakers_in(donor)) <= 1:
                    continue
                for spk in speakers_in(donor):
                    s = len(rows_by_speaker[spk])
                    if sizes[donor] - s >= min_utts and sizes[lo] + s <= max_utts:
                        move(spk, donor, lo)
                        progressed = True
                        break
                if progressed:
                    break
            if progressed:
                break

        if not progressed:
            break
    return assignment, speaker_to_household


# ----------------------------------------------------------------------------- #
# Verification / reporting
# ----------------------------------------------------------------------------- #
def _effective_intents(intent_series: pd.Series) -> float:
    """exp(Shannon entropy) of a household's label distribution."""
    p = intent_series.value_counts(normalize=True).to_numpy()
    p = p[p > 0]
    entropy = -(p * np.log(p)).sum()
    return float(np.exp(entropy))


def verify_partition(
    result: PartitionResult,
    df: pd.DataFrame,
    min_utts: int = MIN_UTTS,
    max_utts: int = MAX_UTTS,
    min_intents: int = MIN_INTENTS,
    max_intents: int = MAX_INTENTS,
) -> dict:
    """Check Week 3 constraints and return a report dict (incl. effective intents)."""
    sizes = np.array([len(rows) for rows in result.assignment.values()])
    n_intents = np.array([df.iloc[rows]["intent"].nunique() for rows in result.assignment.values()])
    eff_intents = np.array(
        [_effective_intents(df.iloc[rows]["intent"]) for rows in result.assignment.values()]
    )

    all_rows = [r for rows in result.assignment.values() for r in rows]
    coverage_ok = sorted(all_rows) == list(range(len(df)))

    atomic_ok = True
    for _w, idx in df.groupby("worker_id").indices.items():
        homes = {h for h, rows in result.assignment.items() if set(int(i) for i in idx) & set(rows)}
        if len(homes) != 1:
            atomic_ok = False
            break

    return {
        "num_households": result.num_households,
        "alpha": result.alpha,
        "seed": result.seed,
        "total_utts": int(sizes.sum()),
        "utts_min": int(sizes.min()),
        "utts_median": int(np.median(sizes)),
        "utts_max": int(sizes.max()),
        "intents_min": int(n_intents.min()),
        "intents_median": int(np.median(n_intents)),
        "intents_max": int(n_intents.max()),
        "eff_intents_min": round(float(eff_intents.min()), 2),
        "eff_intents_median": round(float(np.median(eff_intents)), 2),
        "eff_intents_max": round(float(eff_intents.max()), 2),
        "size_ok": bool((sizes >= min_utts).all() and (sizes <= max_utts).all()),
        "intents_ok": bool((n_intents >= min_intents).all() and (n_intents <= max_intents).all()),
        "eff_intents_ok": bool(
            (eff_intents >= min_intents).all() and (eff_intents <= max_intents).all()
        ),
        "coverage_ok": bool(coverage_ok),
        "atomic_ok": bool(atomic_ok),
        "n_size_violations": int(((sizes < min_utts) | (sizes > max_utts)).sum()),
        "n_intent_violations": int(((n_intents < min_intents) | (n_intents > max_intents)).sum()),
        "n_eff_intent_violations": int(
            ((eff_intents < min_intents) | (eff_intents > max_intents)).sum()
        ),
    }


def household_label_matrix(
    result: PartitionResult,
    df: pd.DataFrame,
    label2id: dict[str, int] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return an (num_households x num_intents) count matrix + intent names."""
    if label2id is None:
        names = sorted(df["intent"].unique())
        label2id = {n: i for i, n in enumerate(names)}
    else:
        names = [n for n, _ in sorted(label2id.items(), key=lambda kv: kv[1])]

    mat = np.zeros((result.num_households, len(label2id)), dtype=int)
    for h, rows in result.assignment.items():
        vc = df.iloc[rows]["intent"].value_counts()
        for intent, count in vc.items():
            mat[h, label2id[intent]] = count
    return mat, names


# ----------------------------------------------------------------------------- #
# Persistence  (the byte-identical artifact)
# ----------------------------------------------------------------------------- #
def to_dataframe(result: PartitionResult, df: pd.DataFrame) -> pd.DataFrame:
    """Flatten the partition into a tidy, sorted long-form DataFrame."""
    out = []
    for h in range(result.num_households):
        for r in result.assignment[h]:
            rec = df.iloc[r]
            out.append(
                {
                    "household_id": h,
                    "row_index": int(r),
                    "worker_id": rec["worker_id"],
                    "intent": rec["intent"],
                    "utt": rec["utt"],
                }
            )
    part = pd.DataFrame.from_records(
        out, columns=["household_id", "row_index", "worker_id", "intent", "utt"]
    )
    return part.sort_values(["household_id", "row_index"]).reset_index(drop=True)


def save_partition(result: PartitionResult, df: pd.DataFrame, path: str | Path) -> Path:
    """Write the partition to parquet (deterministically) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = to_dataframe(result, df)
    part = part.astype(
        {
            "household_id": "int32",
            "row_index": "int64",
            "worker_id": "string",
            "intent": "string",
            "utt": "string",
        }
    )
    part.to_parquet(path, index=False, engine="pyarrow")
    return path


def load_partition(path: str | Path) -> pd.DataFrame:
    """Read a saved partition parquet back into a long-form DataFrame."""
    return pd.read_parquet(path, engine="pyarrow")
