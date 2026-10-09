"""fl/params.py -- the transmit contract.

The bug these defend against is silent: a get/set ordering mismatch corrupts
weights without raising, and the run still produces a plausible accuracy curve.
So the round-trip test asserts on tensor *values*, not just on shapes.

Fast tests need no torch. Slow tests build a real PEFT model.
"""

import numpy as np
import pytest

from fedknob.fl import params as P

# ---------------------------------------------------------------------------
# fast -- no torch, no model
# ---------------------------------------------------------------------------


def test_expected_counts_are_derived_not_guessed():
    """26 = 12 lora_A + 12 lora_B + classifier weight+bias; split_ab drops the B's."""
    n_layers, n_targets = 6, 2  # 6 DistilBERT layers x (q_lin, v_lin)
    n_a = n_b = n_layers * n_targets
    head = 2  # classifier.weight, classifier.bias
    assert P.EXPECTED_KEYS[P.MODE_EPHEMERAL] == n_a + n_b + head == 26
    assert P.EXPECTED_KEYS[P.MODE_SPLIT_AB] == n_a + head == 14
    # the difference is exactly the per-client state
    assert (P.EXPECTED_KEYS[P.MODE_EPHEMERAL] - P.EXPECTED_KEYS[P.MODE_SPLIT_AB]) == n_b == 12


def test_unknown_mode_rejected():
    for bad in ("split-ab", "SPLIT_AB", "fedavg", ""):
        with pytest.raises(ValueError, match="unknown personalisation mode"):
            P.transmit_keys(_FakeModel(), bad)


# ---------------------------------------------------------------------------
# init arms -- which checkpoint feeds round 0
# ---------------------------------------------------------------------------


def test_the_two_arms_and_their_checkpoints():
    assert sorted(P.INIT_ARMS) == ["proxy", "random"]
    assert P.INIT_ARMS["proxy"] == (P.PROXY_CHECKPOINT, True)
    # random must not warm-start; that is the only thing separating it from proxy
    assert P.INIT_ARMS["random"][1] is False


def test_proxy_and_default_are_different_checkpoints():
    """If these ever collide, the proxy arm silently becomes the centralized arm and
    every downstream 'the server never saw client data' claim is false while
    every test still passes."""
    assert P.PROXY_CHECKPOINT != P.DEFAULT_CHECKPOINT


def test_warm_start_default_is_the_centralized_checkpoint():
    """Pins the no-argument call: any caller that omits
    the checkpoint must keep reproducing the published warm curve."""
    import inspect

    sig = inspect.signature(P.warm_start)
    assert sig.parameters["checkpoint_dir"].default == P.DEFAULT_CHECKPOINT


class _FakeParam:
    def __init__(self, name, shape, trainable):
        self.requires_grad = trainable
        self._shape = shape

    def numel(self):
        return int(np.prod(self._shape))


class _FakeModel:
    """Mimics the frozen-head regime's named_parameters() without torch."""

    def __init__(self):
        self._p = {}
        for layer in range(6):
            for tgt in ("q_lin", "v_lin"):
                base = f"base_model.model.distilbert.transformer.layer.{layer}.attention.{tgt}"
                self._p[f"{base}.lora_A.default.weight"] = _FakeParam("", (8, 768), True)
                self._p[f"{base}.lora_B.default.weight"] = _FakeParam("", (768, 8), True)
        self._p["base_model.model.classifier.modules_to_save.default.weight"] = _FakeParam(
            "", (60, 768), True
        )
        self._p["base_model.model.classifier.modules_to_save.default.bias"] = _FakeParam(
            "", (60,), True
        )
        # frozen -- must never appear in either transmit set
        self._p["base_model.model.pre_classifier.weight"] = _FakeParam("", (768, 768), False)
        self._p["base_model.model.pre_classifier.bias"] = _FakeParam("", (768,), False)

    def named_parameters(self):
        return list(self._p.items())


def test_key_counts_on_a_model_shaped_like_ours():
    m = _FakeModel()
    assert len(P.verify_transmit_keys(m, P.MODE_EPHEMERAL)) == 26
    assert len(P.verify_transmit_keys(m, P.MODE_SPLIT_AB)) == 14


def test_frozen_pre_classifier_never_transmitted():
    """It is frozen, so it must not appear in either mode's wire format."""
    m = _FakeModel()
    for mode in P.MODES:
        assert not any("pre_classifier" in k for k in P.transmit_keys(m, mode))


def test_split_ab_withholds_exactly_the_b_matrices():
    m = _FakeModel()
    eph = set(P.transmit_keys(m, P.MODE_EPHEMERAL))
    split = set(P.transmit_keys(m, P.MODE_SPLIT_AB))
    withheld = eph - split
    assert len(withheld) == 12
    assert all("lora_B" in k for k in withheld)
    assert P.local_keys(m, P.MODE_SPLIT_AB) == sorted(withheld)
    # under ephemeral nothing is kept back -- that is what makes it vanilla FedAvg
    assert P.local_keys(m, P.MODE_EPHEMERAL) == []


def test_key_order_is_sorted_and_therefore_reproducible():
    """named_parameters() order is an implementation detail; the wire format
    must not depend on it."""
    m = _FakeModel()
    keys = P.transmit_keys(m, P.MODE_EPHEMERAL)
    assert keys == sorted(keys)
    assert keys == P.transmit_keys(_FakeModel(), P.MODE_EPHEMERAL)


def test_arity_assert_fires_on_a_missing_trainable_tensor():
    """Drop a TRAINABLE tensor -- dropping a frozen one must not change the
    transmit set, which is itself the point of the previous test."""
    m = _FakeModel()
    victim = next(k for k, p in m._p.items() if p.requires_grad and "lora_A" in k)
    del m._p[victim]
    with pytest.raises(AssertionError, match="transmit key count"):
        P.verify_transmit_keys(m, P.MODE_EPHEMERAL)


def test_dropping_a_frozen_tensor_does_not_change_the_transmit_set():
    m = _FakeModel()
    before = P.transmit_keys(m, P.MODE_EPHEMERAL)
    del m._p["base_model.model.pre_classifier.bias"]
    assert P.transmit_keys(m, P.MODE_EPHEMERAL) == before


def test_split_ab_assert_fires_if_lora_b_leaks_in(monkeypatch):
    """If LOCAL_MARKER ever stops matching PEFT's naming, split_ab would silently
    transmit B and personalisation would vanish. That must fail loudly."""
    m = _FakeModel()
    monkeypatch.setattr(P, "LOCAL_MARKER", "nonexistent_marker")
    with pytest.raises(AssertionError, match="transmit key count"):
        P.verify_transmit_keys(m, P.MODE_SPLIT_AB)


def test_params_summary_counts_match_the_pinned_totals():
    s = P.params_summary(_FakeModel(), P.MODE_EPHEMERAL)
    assert s["transmit_keys"] == 26
    assert s["local_keys"] == 0
    assert s["trainable_params"] == 193_596  # matches distilbert_lora.py
    s2 = P.params_summary(_FakeModel(), P.MODE_SPLIT_AB)
    assert s2["local_params"] == 73_728  # LoRA B -- the sole per-client state
    assert s2["transmit_params"] + s2["local_params"] == 193_596


# ---------------------------------------------------------------------------
# slow -- builds a real PEFT model
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_model_key_counts(head_checkpoint):
    from fedknob.models.distilbert_lora import build_lora_model

    m = build_lora_model(num_labels=60, freeze_pre_classifier=True)
    assert len(P.verify_transmit_keys(m, P.MODE_EPHEMERAL)) == 26
    assert len(P.verify_transmit_keys(m, P.MODE_SPLIT_AB)) == 14
    assert P.params_summary(m, P.MODE_EPHEMERAL)["trainable_params"] == 193_596


@pytest.mark.slow
@pytest.mark.parametrize("mode", [P.MODE_EPHEMERAL, P.MODE_SPLIT_AB])
def test_round_trip_preserves_values_exactly(mode, head_checkpoint):
    """set(get(m)) must be the identity. This is the check that catches a
    key-ordering bug -- shapes alone would not, since many tensors share one."""
    from fedknob.models.distilbert_lora import build_lora_model

    m = build_lora_model(num_labels=60, freeze_pre_classifier=True)
    before = P.get_params(m, mode)
    P.set_params(m, before, mode)
    after = P.get_params(m, mode)
    assert len(before) == len(after) == P.EXPECTED_KEYS[mode]
    for b, a in zip(before, after, strict=True):
        np.testing.assert_array_equal(b, a)


@pytest.mark.slow
def test_set_params_rejects_wrong_length(head_checkpoint):
    from fedknob.models.distilbert_lora import build_lora_model

    m = build_lora_model(num_labels=60, freeze_pre_classifier=True)
    arrays = P.get_params(m, P.MODE_EPHEMERAL)
    with pytest.raises(ValueError, match="disagree about what is transmitted"):
        P.set_params(m, arrays[:-1], P.MODE_EPHEMERAL)


@pytest.mark.slow
def test_get_params_returns_snapshots_not_views(head_checkpoint):
    """``Tensor.numpy()`` aliases the tensor's storage and ``load_state_dict``
    writes in place. Without a copy, arrays already handed to the server mutate
    as the local model keeps training -- FedAvg would then average post-training
    weights it believes are pre-training, with every shape and count check still
    passing. This pins the copy."""
    from fedknob.models.distilbert_lora import build_lora_model

    m = build_lora_model(num_labels=60, freeze_pre_classifier=True)
    snapshot = P.get_params(m, P.MODE_EPHEMERAL)
    frozen_copy = [a.copy() for a in snapshot]
    P.set_params(m, [a + 1.0 for a in snapshot], P.MODE_EPHEMERAL)
    for taken, expected in zip(snapshot, frozen_copy, strict=True):
        np.testing.assert_array_equal(
            taken, expected, err_msg="get_params returned a view into the model, not a snapshot"
        )


@pytest.mark.slow
def test_set_params_actually_writes(head_checkpoint):
    """A no-op set_params would make every round of FedAvg silently pointless."""
    from fedknob.models.distilbert_lora import build_lora_model

    m = build_lora_model(num_labels=60, freeze_pre_classifier=True)
    orig = P.get_params(m, P.MODE_EPHEMERAL)
    P.set_params(m, [a + 1.0 for a in orig], P.MODE_EPHEMERAL)
    now = P.get_params(m, P.MODE_EPHEMERAL)
    for o, n in zip(orig, now, strict=True):
        np.testing.assert_allclose(n, o + 1.0, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# arm A2 -- model replacement (fast, no torch)
# ---------------------------------------------------------------------------


def test_update_norm_is_the_l2_of_the_flattened_delta():
    snap = [np.zeros((2, 2)), np.zeros(1)]
    local = [np.array([[3.0, 0.0], [0.0, 4.0]]), np.zeros(1)]
    assert P.update_norm(snap, local) == pytest.approx(5.0)
    assert P.update_norm(snap, snap) == 0.0


# ---------------------------------------------------------------------------
# where the global update lands -- LoRA (attention) vs classifier (scoring)
# ---------------------------------------------------------------------------

KEYS = [
    "distilbert.transformer.layer.0.attention.q_lin.lora_A.default.weight",
    "distilbert.transformer.layer.0.attention.v_lin.lora_B.default.weight",
    "classifier.modules_to_save.default.weight",
    "classifier.modules_to_save.default.bias",
]


def test_group_of_survives_peft_naming():
    # PEFT writes `...lora_A.default.weight` and modules_to_save renames the head
    # to `classifier.modules_to_save.default.*`. Substring matching survives a
    # PEFT upgrade; positional assumptions do not.
    assert [P.group_of(k) for k in KEYS] == ["lora", "lora", "classifier", "classifier"]
    assert P.group_of("distilbert.embeddings.word_embeddings.weight") == "other"


def test_classifier_only_movement_reads_as_prior_shift():
    # Movement that reached the output layer and nothing else, so
    # it can only raise one class's logit for every input.
    before = [np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(2)]
    after = [np.zeros(4), np.zeros(4), np.array([3.0, 4.0, 0.0, 0.0]), np.zeros(2)]
    r = P.grouped_delta_norms(before, after, KEYS)
    assert r["global_classifier_delta"] == pytest.approx(5.0)
    assert r["global_lora_delta"] == 0.0
    assert r["global_lora_frac"] == 0.0


def test_lora_only_movement_reads_as_attention_change():
    before = [np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(2)]
    after = [np.array([3.0, 4.0, 0.0, 0.0]), np.zeros(4), np.zeros(4), np.zeros(2)]
    r = P.grouped_delta_norms(before, after, KEYS)
    assert r["global_lora_delta"] == pytest.approx(5.0)
    assert r["global_lora_frac"] == 1.0


def test_no_movement_does_not_divide_by_zero():
    z = [np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(2)]
    assert P.grouped_delta_norms(z, z, KEYS)["global_lora_frac"] == 0.0
