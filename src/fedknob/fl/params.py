"""Which tensors leave a client, and in what order.

Flower hands a client a bare ``list[np.ndarray]`` -- no names, no dict. The
ordering is the entire contract: if ``get_params`` and ``set_params`` disagree by
one position, weights land in the wrong tensors, no exception is raised, and the
run produces plausible-looking garbage. Every guard in this module exists for
that one failure.

Two personalisation modes, and the only difference between them is which keys
are in the list:

    ephemeral   LoRA A + LoRA B + classifier      26 tensors   plain FedAvg
    split_ab    LoRA A          + classifier      14 tensors   B stays local

``split_ab`` is not implemented by subclassing a Flower strategy. It is
implemented here, by returning fewer arrays. Flower averages whatever it is
given, so withholding ``lora_B`` is sufficient to keep it per-client.

Key names are derived from ``requires_grad`` rather than hardcoded. A live PEFT
model names its tensors differently from the saved adapter file --
``lora_A.default.weight`` and ``classifier.modules_to_save.default.weight``
in the model, ``lora_A.weight`` and ``classifier.weight`` on disk -- so any
hardcoded list is wrong in one of the two contexts. Deriving from the trainable
set is correct in both, and self-validates: in the frozen-head regime the
trainable tensors ARE exactly the 26 we mean to transmit.
"""

from __future__ import annotations

from pathlib import Path

MODE_EPHEMERAL = "ephemeral"
MODE_SPLIT_AB = "split_ab"
MODES = (MODE_EPHEMERAL, MODE_SPLIT_AB)

#: Expected tensor counts, freeze_pre_classifier=True.
#:   12 lora_A (q_lin, v_lin x 6 layers) + 12 lora_B + classifier.{weight,bias}
#: Pinned so a PEFT rename or an unfrozen module fails on run 1 rather than
#: after a sweep. With pre_classifier trainable these become 28 and 16.
EXPECTED_KEYS = {MODE_EPHEMERAL: 26, MODE_SPLIT_AB: 14}

#: Substring identifying the per-client tensors withheld under split_ab.
LOCAL_MARKER = "lora_B"


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"unknown personalisation mode {mode!r}; expected one of {MODES}")


def transmit_keys(model, mode: str = MODE_EPHEMERAL) -> list[str]:
    """Return the state-dict keys this client uploads, in a stable order.

    Sorted, so the order is identical across processes, PEFT versions and
    machines -- ``named_parameters()`` order is an implementation detail and is
    not guaranteed stable enough to be a wire format.
    """
    _check_mode(mode)
    keys = [n for n, p in model.named_parameters() if p.requires_grad]
    if mode == MODE_SPLIT_AB:
        keys = [k for k in keys if LOCAL_MARKER not in k]
    return sorted(keys)


def verify_transmit_keys(model, mode: str = MODE_EPHEMERAL) -> list[str]:
    """``transmit_keys`` plus the arity assert. Call once at construction."""
    keys = transmit_keys(model, mode)
    expected = EXPECTED_KEYS[mode]
    if len(keys) != expected:
        raise AssertionError(
            f"transmit key count {len(keys)} != expected {expected} for mode "
            f"{mode!r}.\nGot: {keys}\nA PEFT rename, an unfrozen module, or "
            f"freeze_pre_classifier=False will show up here first."
        )
    if mode == MODE_SPLIT_AB and any(LOCAL_MARKER in k for k in keys):
        raise AssertionError(
            f"{LOCAL_MARKER} present in the split_ab transmit set -- it would be "
            f"aggregated, which silently removes all personalisation."
        )
    return keys


def get_params(model, mode: str = MODE_EPHEMERAL) -> list:
    """Trainable tensors as numpy arrays, ordered by ``transmit_keys``.

    The ``.copy()`` is load-bearing. ``Tensor.numpy()`` returns a view sharing
    storage with the tensor, and ``load_state_dict`` writes in place -- so
    without it, arrays handed to the server keep changing as the local model
    trains. A snapshot taken before a round would silently become the weights
    from after it, and FedAvg would average the wrong thing while every shape
    and count check still passed.
    """
    sd = model.state_dict()
    return [sd[k].detach().cpu().numpy().copy() for k in transmit_keys(model, mode)]


def set_params(model, arrays, mode: str = MODE_EPHEMERAL) -> None:
    """Write ``arrays`` back, in the same order ``get_params`` produced them.

    Raises on a length or shape mismatch. Without these checks a wrong-length
    list silently sets a prefix of the tensors, which is the failure this whole
    module is defending against.
    """
    import torch

    keys = transmit_keys(model, mode)
    if len(arrays) != len(keys):
        raise ValueError(
            f"got {len(arrays)} arrays for {len(keys)} keys (mode={mode!r}). "
            f"The sender and receiver disagree about what is transmitted."
        )
    sd = model.state_dict()
    updates = {}
    for k, arr in zip(keys, arrays, strict=True):
        want = tuple(sd[k].shape)
        got = tuple(arr.shape)
        if want != got:
            raise ValueError(f"shape mismatch for {k}: expected {want}, got {got}")
        updates[k] = torch.as_tensor(arr, dtype=sd[k].dtype)
    model.load_state_dict(updates, strict=False)


#: Week 2 adapter, fitted on the TRAINING split -- which is also the split the
#: federated clients are partitioned from. It has therefore already seen every
#: client's private rows. Legitimate as an upper-bound arm ("what if the server
#: had centralised everything"), but it is not a valid FL initialisation: a
#: server able to fit that data would have no reason to federate.
DEFAULT_CHECKPOINT = "artifacts/baseline/distilbert_lora_r8"

#: Server-side proxy init, fitted on the held-out validation split only. It has
#: seen none of the clients' rows, so it is what a real deployment could hold at
#: round 0. Weaker standalone -- that gap is the headroom the rounds must close.
PROXY_CHECKPOINT = "artifacts/baseline/distilbert_lora_r8_server_init"

#: init arm -> (checkpoint supplying the frozen ``pre_classifier``, warm-start?).
#: ``random`` keeps DEFAULT_CHECKPOINT so the published cold curve reproduces
#: exactly; note that this leaves its 590,592 frozen head train-derived. For a
#: cold arm with no client exposure at all, pass ``--checkpoint`` explicitly.
INIT_ARMS: dict[str, tuple[str, bool]] = {
    "week2": (DEFAULT_CHECKPOINT, True),
    "proxy": (PROXY_CHECKPOINT, True),
    "random": (DEFAULT_CHECKPOINT, False),
}


def warm_start(model, checkpoint_dir: str | Path = DEFAULT_CHECKPOINT) -> int:
    """Load the Week-2 LoRA A/B and classifier into ``model``. Returns keys set.

    This is the ``init.classifier="week2"`` arm. A cold start skips it entirely
    and leaves ``classifier`` at its random initialisation and ``B`` at zero.

    The saved adapter and the live model name the same tensors differently --
    ``lora_A.weight`` on disk against ``lora_A.default.weight`` in the model, and
    ``classifier.weight`` against ``classifier.modules_to_save.default.weight``.
    ``set_peft_model_state_dict`` owns that mapping, so we do not reimplement it;
    a hand-rolled mapping is exactly the kind of thing that silently half-works
    after a PEFT upgrade.

    Note this also rewrites ``pre_classifier`` with the same values
    ``build_lora_model(freeze_pre_classifier=True)`` already loaded -- a no-op,
    but harmless and it keeps the checkpoint the single source of truth.
    """
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    from fedknob.data.massive import PROJECT_ROOT

    path = Path(checkpoint_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    weights = path / "adapter_model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(
            f"warm-start checkpoint not found: {weights}\n"
            f"Run scripts/train_lora_massive.py first, or use a cold start."
        )

    state = load_file(str(weights))
    result = set_peft_model_state_dict(model, state)
    missing = getattr(result, "unexpected_keys", None)
    if missing:
        raise RuntimeError(
            f"warm start left {len(missing)} unexpected key(s) unmapped: "
            f"{list(missing)[:5]}. PEFT's key mapping and this checkpoint disagree."
        )
    return len(state)


def local_keys(model, mode: str = MODE_EPHEMERAL) -> list[str]:
    """Trainable tensors that are NOT transmitted -- the per-client state.

    Empty under ``ephemeral``; the 12 ``lora_B`` tensors under ``split_ab``.
    This is the set ``fl/state.py`` persists to disk, and under split_ab it is
    the only thing a client carries between rounds.
    """
    _check_mode(mode)
    sent = set(transmit_keys(model, mode))
    return sorted(n for n, p in model.named_parameters() if p.requires_grad and n not in sent)


def scale_update(snapshot: list, local: list, gamma: float) -> list:
    """Model replacement -- return ``snapshot + gamma * (local - snapshot)``.

    Arm A2 (Bagdasaryan et al. 2020 3). FedAvg weights a client's contribution by
    ``n_i / N``, so on this federation an attacker holding ~26 of a round's ~990
    rows has ~97% of its update averaged away. Pre-multiplying the *delta* by
    ``gamma ~ N / n_i`` cancels that weight, and at ``gamma = N/n_i`` the global
    model after aggregation is approximately the attacker's own model -- hence
    "replacement" rather than "influence".

    **Scale the delta, never the weights.** ``gamma * local`` would multiply the
    absolute parameter values, which is a different and much more destructive
    operation: it moves every coordinate away from the origin rather than moving
    along the direction local training actually travelled.

    ``gamma = 1`` returns ``local`` exactly (up to float error), which is what
    makes arm A1 a special case of this function rather than a separate path --
    two code paths is how an A1 run ends up labelled A2.

    THE OVERSHOOT THIS DOES NOT PROTECT AGAINST
    -------------------------------------------
    Bagdasaryan assume **one** attacker per round. Under natural sampling this
    federation draws ~3.8, and if each scales by the full ``N/n_i`` the aggregate
    moves ~3.8x further along the attack direction than any of them intended.
    ``delta`` is the endpoint of a few epochs of local SGD, not a ray that can be
    extended freely: walking 3.8x along it lands outside the region local
    training explored, and the usual result is a model that is bad at everything
    including the backdoor. Choosing gamma is the caller's problem; see
    the attack phase notes.
    """
    import numpy as np

    if gamma < 0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")
    if len(snapshot) != len(local):
        raise ValueError(
            f"snapshot has {len(snapshot)} arrays, local has {len(local)} -- "
            f"these must be the same transmit set in the same order."
        )
    out = []
    for s, curr in zip(snapshot, local, strict=True):
        s = np.asarray(s)
        curr = np.asarray(curr)
        if s.shape != curr.shape:
            raise ValueError(f"shape mismatch in scale_update: {s.shape} vs {curr.shape}")
        out.append(s + gamma * (curr - s))
    return out


def update_norm(snapshot: list, local: list) -> float:
    """L2 norm of the flattened update ``local - snapshot``.

    Recorded every round for three reasons, all of which want the *same* number:
    diagnosing a diverged gamma before 30 rounds of NaN, feeding the norm-clipping
    defense later, and reporting detectability -- a scaled update's norm is
    precisely the signal Sun et al. (arXiv:1911.07963) clip against, so an attack
    that needs large gamma is declaring how easy it is to catch.
    """
    import numpy as np

    total = 0.0
    for s, curr in zip(snapshot, local, strict=True):
        d = np.asarray(curr, dtype=np.float64) - np.asarray(s, dtype=np.float64)
        total += float((d * d).sum())
    return float(np.sqrt(total))


#: Which transmitted tensors belong to which functional block. Used to split a
#: global update into "what the model attends to" versus "how it scores".
#:
#: The split is load-bearing for reading a backdoor. The classifier is a *linear
#: map on a fixed representation*: if the pooled [CLS] vector does not change
#: when the trigger is present, no setting of the classifier weights can make the
#: prediction depend on the trigger. All it can do is raise one class's logit for
#: everything -- a prior shift. A conditional trigger->target rule therefore has
#: to live in the LoRA adapters on ``q_lin``/``v_lin``, which are the only
#: trainable tensors that can change *what [CLS] attends to*.
#:
#: So the two groups distinguish the two failure modes observed in Phase I: an
#: attack that moves only ``classifier`` inflates ``asr_base`` alongside ``asr``
#: and leaves delta_ASR at zero, while an attack that moves ``lora`` is the one
#: that produces a real trigger association.
PARAM_GROUPS = {"lora": ("lora_A", "lora_B"), "classifier": ("classifier",)}


def group_of(key: str) -> str:
    """Which ``PARAM_GROUPS`` block a transmit key belongs to.

    ``modules_to_save`` renames the head to ``classifier.modules_to_save.default.*``
    and PEFT names adapters ``...lora_A.default.weight``, so substring matching is
    what survives a PEFT upgrade; positional assumptions do not.
    """
    for g, marks in PARAM_GROUPS.items():
        if any(m in key for m in marks):
            return g
    return "other"


def grouped_delta_norms(before: list, after: list, keys: list[str]) -> dict:
    """Per-group L2 norm of ``after - before``, plus the LoRA share.

    Called on the **global** model between consecutive rounds, not on a client's
    update. That distinction is the whole point: every poisoned client update
    contains both LoRA and classifier changes in a fixed ratio -- gamma scales
    them uniformly -- so a client-side split cannot distinguish arm A1 from A2.
    What differs between the arms is which of those changes *survives averaging*,
    and only the aggregated model shows that.
    """
    import numpy as np

    tot = dict.fromkeys(PARAM_GROUPS, 0.0)
    for k, b, a in zip(keys, before, after, strict=True):
        g = group_of(k)
        if g in tot:
            d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
            tot[g] += float((d * d).sum())
    out = {f"global_{g}_delta": float(np.sqrt(v)) for g, v in tot.items()}
    denom = out["global_lora_delta"] + out["global_classifier_delta"]
    # Share of the round's global movement that landed in attention rather than
    # in the output layer. The round this starts rising should be the round
    # delta_ASR crosses zero; if it is not, the mechanism story is wrong.
    out["global_lora_frac"] = out["global_lora_delta"] / denom if denom else 0.0
    return out


def params_summary(model, mode: str = MODE_EPHEMERAL) -> dict:
    """Counts for the run manifest -- what moved, what stayed, how big."""
    sent, kept = transmit_keys(model, mode), local_keys(model, mode)
    by_name = dict(model.named_parameters())
    n_sent = sum(by_name[k].numel() for k in sent)
    n_kept = sum(by_name[k].numel() for k in kept)
    return {
        "mode": mode,
        "transmit_keys": len(sent),
        "transmit_params": n_sent,
        "local_keys": len(kept),
        "local_params": n_kept,
        "trainable_params": n_sent + n_kept,
        "transmit_kb_fp32": round(n_sent * 4 / 1024, 1),
    }
