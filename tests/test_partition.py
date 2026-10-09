"""Partition tests: determinism, speaker-atomicity, constraints, skew.

These run on a small synthetic MASSIVE-shaped fixture so they need no network
and finish in milliseconds; the real-data exit-criterion check lives in
scripts/partition_households.py --check-byte-identical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fedknob.data.partition import (
    household_label_matrix,
    partition_by_speaker_dirichlet,
    save_partition,
    to_dataframe,
    verify_partition,
)


def _fixture(n_speakers: int = 120, n_intents: int = 60, seed: int = 0) -> pd.DataFrame:
    """A MASSIVE-shaped frame: worker_id, intent, utt. Each speaker has a
    dominant intent plus a little spillover, mimicking real annotator skew."""
    rng = np.random.default_rng(seed)
    rows = []
    intents = [f"intent_{i:02d}" for i in range(n_intents)]
    for s in range(n_speakers):
        worker = f"w{s:04d}"
        dominant = intents[s % n_intents]
        n = int(rng.integers(8, 25))
        for j in range(n):
            intent = dominant if rng.random() < 0.8 else intents[rng.integers(n_intents)]
            rows.append({"worker_id": worker, "intent": intent,
                         "utt": f"{worker} utterance {j}"})
    return pd.DataFrame(rows)


def test_determinism_same_seed_same_partition():
    df = _fixture()
    a = partition_by_speaker_dirichlet(df, num_households=20, alpha=0.5, seed=42)
    b = partition_by_speaker_dirichlet(df, num_households=20, alpha=0.5, seed=42)
    assert a.assignment == b.assignment
    assert a.speaker_to_household == b.speaker_to_household


def test_byte_identical_parquet(tmp_path):
    df = _fixture()
    a = partition_by_speaker_dirichlet(df, num_households=20, seed=42)
    b = partition_by_speaker_dirichlet(df, num_households=20, seed=42)
    pa, pb = tmp_path / "a.parquet", tmp_path / "b.parquet"
    save_partition(a, df, pa)
    save_partition(b, df, pb)
    assert pa.read_bytes() == pb.read_bytes()


def test_different_seed_changes_partition():
    df = _fixture()
    a = partition_by_speaker_dirichlet(df, num_households=20, seed=42)
    b = partition_by_speaker_dirichlet(df, num_households=20, seed=7)
    assert a.assignment != b.assignment


def test_speaker_atomicity_and_full_coverage():
    df = _fixture()
    res = partition_by_speaker_dirichlet(df, num_households=20, seed=42)
    rep = verify_partition(res, df)
    assert rep["coverage_ok"]   # every utterance assigned exactly once
    assert rep["atomic_ok"]     # each speaker in exactly one household


def test_size_repair_respects_bounds_when_feasible():
    df = _fixture(n_speakers=200)
    res = partition_by_speaker_dirichlet(
        df, num_households=20, seed=42, min_utts=30, max_utts=300
    )
    rep = verify_partition(res, df, min_utts=30, max_utts=300)
    assert rep["utts_min"] >= 30
    assert rep["utts_max"] <= 300


def test_too_few_speakers_raises():
    df = _fixture(n_speakers=10)
    with pytest.raises(ValueError):
        partition_by_speaker_dirichlet(df, num_households=20)


def test_label_matrix_shape_and_totals():
    df = _fixture()
    res = partition_by_speaker_dirichlet(df, num_households=20, seed=42)
    mat, names = household_label_matrix(res, df)
    assert mat.shape == (20, df["intent"].nunique())
    assert mat.sum() == len(df)          # all utterances accounted for
    assert len(names) == df["intent"].nunique()


def test_dirichlet_produces_skew():
    """A small alpha should give a markedly more skewed per-household label
    distribution than a large one (sanity check on the mechanism)."""
    df = _fixture(n_speakers=200)
    skewed = partition_by_speaker_dirichlet(df, num_households=20, alpha=0.1, seed=1)
    uniform = partition_by_speaker_dirichlet(df, num_households=20, alpha=100.0, seed=1)

    def mean_top_share(res):
        mat, _ = household_label_matrix(res, df)
        shares = mat.max(axis=1) / mat.sum(axis=1).clip(min=1)
        return shares.mean()

    assert mean_top_share(skewed) > mean_top_share(uniform)


def test_effective_intents_reported_and_bounded():
    """verify_partition reports effective-intent stats, and the effective count
    never exceeds the raw distinct count (perplexity <= support)."""
    df = _fixture(n_speakers=200)
    res = partition_by_speaker_dirichlet(df, num_households=20, seed=42)
    rep = verify_partition(res, df)
    for key in ("eff_intents_min", "eff_intents_median", "eff_intents_max",
                "eff_intents_ok", "n_eff_intent_violations"):
        assert key in rep
    # effective intents (exp-entropy) is bounded above by raw distinct support
    assert rep["eff_intents_max"] <= rep["intents_max"] + 1e-6
    assert rep["eff_intents_min"] >= 1.0


def test_long_form_columns_and_sorting():
    df = _fixture()
    res = partition_by_speaker_dirichlet(df, num_households=20, seed=42)
    long = to_dataframe(res, df)
    assert list(long.columns) == ["household_id", "row_index", "worker_id",
                                  "intent", "utt"]
    # sorted by (household_id, row_index)
    assert long.equals(long.sort_values(["household_id", "row_index"])
                       .reset_index(drop=True))
