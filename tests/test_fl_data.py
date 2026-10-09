"""fl/data.py -- the split and its invariants.

The two failure modes worth defending against both produce plausible accuracy
rather than an error: losing or duplicating utterances in the split, and a
per-client eval set that moves between runs. Both are checked here without
torch, because the pure logic is factored out of the tokenising path.
"""

import numpy as np
import pytest

from fedknob.fl import data as D

# ---------------------------------------------------------------------------
# split_indices -- pure, no torch
# ---------------------------------------------------------------------------


def test_central_eval_gives_every_row_to_training():
    """eval_fraction=0 is the vanilla FedAvg case: nothing is held out."""
    tr, ev = D.split_indices(108, 0.0, seed=42, client_id=7)
    assert len(tr) == 108
    assert len(ev) == 0
    np.testing.assert_array_equal(tr, np.arange(108))


def test_split_is_a_partition_no_loss_no_overlap():
    for n in (11, 50, 108, 581, 936):
        tr, ev = D.split_indices(n, 0.2, seed=42, client_id=3)
        assert len(tr) + len(ev) == n
        assert set(tr).isdisjoint(ev)
        assert sorted([*tr, *ev]) == list(range(n))


def test_split_is_deterministic_across_calls():
    a = D.split_indices(108, 0.2, seed=42, client_id=5)
    b = D.split_indices(108, 0.2, seed=42, client_id=5)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_different_clients_get_different_splits():
    """Seeding with seed+client_id, not seed alone -- otherwise every client
    holds out the same row positions, which correlates their eval noise."""
    ev5 = D.split_indices(108, 0.2, seed=42, client_id=5)[1]
    ev6 = D.split_indices(108, 0.2, seed=42, client_id=6)[1]
    assert not np.array_equal(ev5, ev6)


def test_client_split_does_not_depend_on_the_other_clients():
    """A client's held-out rows must be stable whether it sits in a 20-client or
    a 100-client partition -- the split is a property of the client alone."""
    solo = D.split_indices(108, 0.2, seed=42, client_id=17)
    again = D.split_indices(108, 0.2, seed=42, client_id=17)
    np.testing.assert_array_equal(solo[1], again[1])


def test_eval_size_rounds_to_the_requested_fraction():
    for n, frac, want in [(100, 0.2, 20), (108, 0.2, 22), (10, 0.2, 2), (581, 0.2, 116)]:
        _, ev = D.split_indices(n, frac, seed=42, client_id=0)
        assert len(ev) == want, f"n={n} frac={frac}"


def test_tiny_client_still_gets_at_least_one_eval_row():
    _, ev = D.split_indices(4, 0.2, seed=42, client_id=0)  # round(0.8) == 1
    assert len(ev) == 1


def test_rejects_a_split_that_would_leave_no_training_data():
    with pytest.raises(ValueError, match="leaving no training data"):
        D.split_indices(2, 0.9, seed=42, client_id=0)


def test_rejects_nonsense_fractions_and_empty_clients():
    with pytest.raises(ValueError, match="eval_fraction must be in"):
        D.split_indices(100, 1.5, seed=42, client_id=0)
    with pytest.raises(ValueError, match="no rows"):
        D.split_indices(0, 0.2, seed=42, client_id=0)


# ---------------------------------------------------------------------------
# verify_clients -- the coverage guarantee
# ---------------------------------------------------------------------------


class _C:
    def __init__(self, cid, n_train, n_eval=0, n_speakers=7):
        self.client_id, self.n_train = cid, n_train
        self.n_eval, self.n_speakers = n_eval, n_speakers


def _clients(sizes, evals=None, speakers=None):
    evals = evals or [0] * len(sizes)
    speakers = speakers or [7] * len(sizes)
    return {
        i: _C(i, n, e, s) for i, (n, e, s) in enumerate(zip(sizes, evals, speakers, strict=True))
    }


def test_coverage_holds_for_a_well_formed_federation():
    D.verify_clients(_clients([50, 60, 70]), n_rows=180, n_households=3, eval_fraction=0.0)


def test_lost_utterances_are_caught():
    with pytest.raises(AssertionError, match="lost or duplicated"):
        D.verify_clients(_clients([50, 60, 70]), n_rows=200, n_households=3, eval_fraction=0.0)


def test_client_count_mismatch_is_caught():
    with pytest.raises(AssertionError, match="clients built from"):
        D.verify_clients(_clients([50, 60, 70]), n_rows=180, n_households=4, eval_fraction=0.0)


def test_empty_client_is_caught():
    with pytest.raises(AssertionError, match="no training data"):
        D.verify_clients(_clients([50, 0, 70]), n_rows=120, n_households=3, eval_fraction=0.0)


def test_stray_eval_rows_under_central_evaluation_are_caught():
    with pytest.raises(AssertionError, match="eval_fraction=0 but"):
        D.verify_clients(_clients([50, 60], [0, 5]), n_rows=115, n_households=2, eval_fraction=0.0)


def test_summarise_reports_the_federation_shape():
    s = D.summarise(_clients([50, 108, 936], [10, 22, 187], speakers=[3, 7, 21]))
    assert s["n_clients"] == 3
    assert s["total_train"] == 1094
    assert s["train_per_client"] == {"min": 50, "median": 108, "max": 936}
    assert s["speakers_per_client"] == {"min": 3, "median": 7, "max": 21}
    assert s["eval_per_client_median"] == 22


# ---------------------------------------------------------------------------
# slow -- needs pyarrow, transformers, datasets and the frozen partition
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_loads_the_frozen_n100_partition():
    clients, label2id = D.load_clients("artifacts/partitions/households_100.parquet")
    assert len(clients) == 100
    assert len(label2id) == 60
    assert sum(c.n_train for c in clients.values()) == 11_514  # full coverage
    assert all(c.eval is None for c in clients.values())  # central eval
    s = D.summarise(clients)
    assert s["train_per_client"]["median"] == 108  # matches the stats json


@pytest.mark.slow
def test_label_map_matches_the_week2_checkpoint_ordering():
    """60 intents, and the map is the same object Weeks 1-2 trained against."""
    from fedknob.data.massive import build_label_map, load_massive_en

    label2id, n = D.load_label_map()
    ref, ref_n = build_label_map(load_massive_en()["train"]["label"])
    assert (label2id, n) == (ref, ref_n) and n == 60


@pytest.mark.slow
def test_eval_fraction_holds_out_and_still_covers():
    clients, _ = D.load_clients("artifacts/partitions/households_100.parquet", eval_fraction=0.2)
    assert sum(c.n_train + c.n_eval for c in clients.values()) == 11_514
    assert all(c.eval is not None and c.n_eval > 0 for c in clients.values())


@pytest.mark.slow
def test_central_eval_split_is_the_untouched_test_set():
    ds = D.load_central_eval("test")
    assert len(ds) == 2_974  # MASSIVE en-US test
