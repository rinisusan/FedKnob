"""fl/server.py -- the round alignment of the sampled-client record.

``sampled`` is produced during *fit* and written during *evaluate*, which are
different points in Flower's loop:

    evaluate(0), fit(1), evaluate(1), fit(2), evaluate(2), ...

so ``evaluate(r)`` follows ``fit(r)`` and must record ``fit(r)``'s clients. Round
0 precedes every fit and must record none.

Why this is worth a test rather than a careful read: the whole persistence claim
is which clients trained in which round. Shifted by one, every round is
mislabelled
and nothing fails -- the numbers stay plausible and the conclusion is wrong.

No torch, no Flower, no model: the ordering is exercised by driving the two
module-level hooks directly.
"""

import pytest

from fedknob.fl import server as S


@pytest.fixture(autouse=True)
def clean_module_state():
    """These are module-level, so a leaked value would make the next test pass
    for the wrong reason -- the same class of bug as the LoRA-B leak."""
    S.HISTORY.clear()
    S.LAST_SAMPLED.clear()
    S.RARE_INTENTS.clear()
    yield
    S.HISTORY.clear()
    S.LAST_SAMPLED.clear()
    S.RARE_INTENTS.clear()


def _fit(client_ids):
    """One fit round's aggregation, shaped as Flower delivers it."""
    results = [
        (10, {"client_id": c, "n_steps": 4, "mean_loss": 0.5, "loss_fell": True})
        for c in client_ids
    ]
    return S.aggregate_fit_metrics(results)


def test_fit_publishes_the_sampled_ids():
    out = _fit([7, 3, 91])
    assert out["sampled"] == [3, 7, 91], "sorted, so runs are comparable"
    assert list(S.LAST_SAMPLED) == [3, 7, 91]


def test_round_zero_has_no_sampled_clients():
    """Round 0 is evaluated before any fit. Anything else here would attribute a
    round of training to a model that had not trained."""
    assert list(S.LAST_SAMPLED) == []


def test_each_fit_replaces_rather_than_appends():
    _fit([1, 2])
    _fit([50, 60])
    assert list(S.LAST_SAMPLED) == [
        50,
        60,
    ], "accumulating would make every round look like it trained on every client sampled so far"


def test_evaluate_after_fit_sees_that_fit_not_the_previous_one():
    """The off-by-one this file exists for."""
    _fit([1, 2, 3])
    snapshot_r1 = list(S.LAST_SAMPLED)
    _fit([4, 5, 6])
    snapshot_r2 = list(S.LAST_SAMPLED)
    assert snapshot_r1 == [1, 2, 3] and snapshot_r2 == [4, 5, 6]


def test_history_rows_copy_the_ids_rather_than_alias_them():
    """``server.py`` stores ``list(LAST_SAMPLED)``. Storing the list itself would
    make every history row point at one object that the next round overwrites --
    so all 31 rounds would report the final round's clients, and the artifact
    would look complete."""
    _fit([1, 2])
    row = {"round": 1, "sampled": list(S.LAST_SAMPLED)}
    _fit([9, 9, 9])
    assert row["sampled"] == [1, 2], "the row must be a snapshot, not a view"


def test_aggregate_reports_the_exit_criterion_signal():
    """loss_fell_frac < 1.0 flags a client that trained without its loss falling
    -- the silent no-op failure E7 watches for."""
    results = [
        (10, {"client_id": 0, "n_steps": 4, "mean_loss": 0.5, "loss_fell": True}),
        (10, {"client_id": 1, "n_steps": 4, "mean_loss": 0.5, "loss_fell": False}),
    ]
    assert S.aggregate_fit_metrics(results)["loss_fell_frac"] == 0.5


def test_empty_fit_result_does_not_raise():
    """Flower can deliver zero results if every sampled client fails; that must
    not take the run down."""
    assert S.aggregate_fit_metrics([]) == {}
