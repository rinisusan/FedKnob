"""fl/task.py -- the per-round metrics a backdoor result has to be read against.

Two of these quantities decide whether the Phase I comparison is interpretable at
all, so they are pinned here rather than trusted:

* ``target_recall`` is measured on ~18 of 2,974 test utterances (``iot_wemo_off``
  is 0.61% of the split). Its standard error is near 11 points and it quantises in
  steps of 5.6, so a single round is not readable. The tests below assert that
  ``target_support`` always travels with it -- a recall figure without its n is
  the specific way this metric misleads.
* ``rare_recall`` pools the k rarest intents to get ~200 examples instead. It is
  the comparator the decay claim actually rests on.

The recall maths is pure numpy, so none of this needs torch or a model.
"""

import numpy as np
import pytest

from fedknob.fl import task as T

# ---------------------------------------------------------------------------
# rare_intent_ids -- which classes form the group comparator
# ---------------------------------------------------------------------------


def test_picks_the_k_least_frequent_classes():
    labels = np.array([0] * 50 + [1] * 30 + [2] * 5 + [3] * 2 + [4] * 9)
    assert T.rare_intent_ids(labels, 1) == [3]
    assert T.rare_intent_ids(labels, 3) == [3, 2, 4]


def test_k_larger_than_the_class_count_returns_every_class():
    """Asking for 10 rarest of 4 classes must not raise or pad with junk."""
    labels = np.array([0, 0, 1, 2, 3])
    got = T.rare_intent_ids(labels, 10)
    assert sorted(got) == [0, 1, 2, 3]


def test_derived_from_the_data_not_hardcoded():
    """The rare set is read off the eval split, so a different label map or split
    changes it automatically rather than silently measuring the wrong classes."""
    a = T.rare_intent_ids(np.array([0] * 10 + [1] * 2), 1)
    b = T.rare_intent_ids(np.array([0] * 2 + [1] * 10), 1)
    assert a == [1] and b == [0]


def test_classes_absent_from_the_split_are_not_selected():
    labels = np.array([5, 5, 7, 7, 7])
    assert set(T.rare_intent_ids(labels, 5)).issubset({5, 7})


# ---------------------------------------------------------------------------
# excluding the attack target -- the invariant the comparison rests on
# ---------------------------------------------------------------------------


def test_target_is_dropped_before_the_k_rarest_are_taken():
    """Not filtered out afterwards -- excluded first, so the group still holds k
    intents. Filtering after would silently return k-1 and shrink the support the
    comparison depends on."""
    labels = np.array([0] * 50 + [1] * 9 + [2] * 8 + [3] * 7 + [4] * 6)
    assert T.rare_intent_ids(labels, 3) == [4, 3, 2]
    got = T.rare_intent_ids(labels, 3, exclude=3)
    assert got == [4, 2, 1], got
    assert len(got) == 3, "excluding must not shrink the group"


def test_excluding_a_class_that_is_not_rare_changes_nothing():
    labels = np.array([0] * 50 + [1] * 9 + [2] * 8 + [3] * 7)
    assert T.rare_intent_ids(labels, 2, exclude=0) == T.rare_intent_ids(labels, 2) == [3, 2]


def test_exclude_none_is_the_unfiltered_behaviour():
    labels = np.array([0] * 50 + [1] * 9 + [2] * 8)
    assert T.rare_intent_ids(labels, 2, exclude=None) == T.rare_intent_ids(labels, 2)


def test_the_target_can_never_reach_the_comparator():
    """The failure this prevents is directional and flattering: a backdoor raises
    its own target's recall, so a comparator containing the target would climb
    under attack and the backdoor would look like it *protects* rare intents.

    Swept across k because the old code relied on the target happening to be
    commoner than the k rarest -- true at k=10, false at k=20."""
    labels = np.concatenate([np.full(60 - i, i) for i in range(30)])
    for k in range(1, 30):
        assert 28 not in T.rare_intent_ids(labels, k, exclude=28)


# ---------------------------------------------------------------------------
# _recall -- pooled over a set of classes, with its support
# ---------------------------------------------------------------------------


def test_recall_is_pooled_not_averaged_per_class():
    """A per-class mean would weight a 2-example class as heavily as a 200-example
    one. With supports this uneven that is dominated by whichever tiny class
    happened to flip, so the pooled form is the one that can be read."""
    labels = np.array([1] * 100 + [2] * 2)
    preds = np.array([1] * 100 + [9, 9])  # every rare-class item wrong
    r, n = T._recall(labels, preds, [1, 2])
    assert n == 102
    assert r == pytest.approx(100 / 102)  # pooled, not (1.0 + 0.0) / 2


def test_recall_counts_only_true_members_of_the_class():
    """Recall, not precision: predictions *into* the class from elsewhere must not
    inflate it. A backdoor pushes predictions toward the target, so a precision-
    shaped bug here would read the attack as improved legitimate performance."""
    labels = np.array([1, 1, 1, 5, 5])
    preds = np.array([1, 1, 9, 1, 1])  # two false positives into 1
    r, n = T._recall(labels, preds, [1])
    assert n == 3 and r == pytest.approx(2 / 3)


def test_absent_class_gives_nan_and_zero_support_not_a_crash():
    """A target with no examples in the split must be visible as missing, not
    reported as 0.0 recall -- those mean very different things."""
    r, n = T._recall(np.array([1, 2]), np.array([1, 2]), [28])
    assert n == 0 and np.isnan(r)


def test_perfect_and_zero_recall_are_exact():
    labels = np.array([3, 3, 3])
    assert T._recall(labels, np.array([3, 3, 3]), [3]) == (1.0, 3)
    assert T._recall(labels, np.array([0, 0, 0]), [3]) == (0.0, 3)


# ---------------------------------------------------------------------------
# the constants the attack phase depends on
# ---------------------------------------------------------------------------


def test_default_target_is_the_documented_attack_target():
    """iot_wemo_off, id 28 -- the Phase I attack target. If the
    label map is ever reordered this constant is what silently breaks."""
    assert T.DEFAULT_TARGET_INTENT == 28


def test_rare_k_default_is_wide_enough_to_be_readable():
    """Raised from 10 to 20 after measuring: at k=10 the group held 81 test
    examples and its decay spread 8.6 points across three seeds, which is too
    coarse to compare a backdoor against. k=20 roughly triples the support."""
    assert T.DEFAULT_RARE_K >= 20


# ---------------------------------------------------------------------------
# slow -- the real evaluator against the real test split
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_evaluate_returns_every_key_the_server_records(head_checkpoint):
    from fedknob.fl import data as D
    from fedknob.models.distilbert_lora import build_lora_model

    model = build_lora_model(num_labels=60, freeze_pre_classifier=True)
    m = T.evaluate(model, D.load_central_eval("test"))

    for key in (
        "accuracy",
        "f1_macro",
        "loss",
        "n_examples",
        "n_correct",
        "target_recall",
        "target_support",
        "rare_recall",
        "rare_support",
    ):
        assert key in m, f"server.py records {key!r} and evaluate() no longer returns it"

    assert m["n_examples"] == 2_974
    assert m["loss"] > 0.0, "a hard-coded 0.0 loss is what this replaced"
    # ~18 expected; pinned loosely because it is a property of the split, and the
    # point of the assertion is that it is small enough to need the group.
    assert 0 < m["target_support"] < 40
    assert m["rare_support"] > 4 * m["target_support"], (
        "the rare group exists to give the decay comparison usable statistics; "
        "if it is not several times the target support it is not doing that"
    )
    # At rare_k=20 the group should hold roughly 250 of the 2,974 test
    # utterances. Pinned loosely -- the point is that one example is now worth
    # well under a point, where at k=10 (n=81) it was worth 1.23.
    assert m["rare_support"] > 150, (
        f"rare_support={m['rare_support']} is smaller than rare_k={T.DEFAULT_RARE_K} "
        f"should give; the comparator will not be precise enough to read"
    )
