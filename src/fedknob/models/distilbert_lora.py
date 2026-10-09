"""DistilBERT + LoRA builder for the centralized baseline and the federated phases.

Wraps the same `distilbert-base-uncased` sequence classifier used by the full
fine-tune baseline (distilbert_classifier.py) with PEFT LoRA adapters. This is
the per-client adapter recipe the federated phases persist across rounds, so
the recipe is defined ONCE here and imported everywhere.

LoRA recipe:
    rank r=8, alpha=16, dropout=0.05
    target_modules = ["q_lin", "v_lin"]   (attention query/value projections)

Parameter accounting -- the property that makes a per-household persistent
adapter feasible:

    LoRA A + B  : 2 modules x 6 layers x (8x768 A + 768x8 B) = 147,456
    pre_classifier (768x768 + 768)                           = 590,592
    classifier  (768x60 + 60)                                =  46,140

TWO HEAD REGIMES
----------------
``freeze_pre_classifier=False`` (default, reproduces the 88.06% baseline)
    trainable = 147,456 + 590,592 + 46,140 = 784,188

``freeze_pre_classifier=True``  (federated phases)
    trainable = 147,456 + 46,140 = 193,596

`pre_classifier` is NEWLY INITIALISED by ``from_pretrained(num_labels=...)`` --
`distilbert-base-uncased` ships no classification head -- so freezing it at its
random values leaves an untrained 768->768 projection permanently in the forward
path and destroys accuracy. It must therefore be initialised from a trained
checkpoint FIRST and frozen second, which is what ``head_init_from`` does.

Why freeze at all: in the federated phases `classifier` is global (aggregated)
and only LoRA ``B`` is persisted per client. Freezing `pre_classifier` leaves
LoRA ``B`` -- 73,728 parameters -- as the *sole* per-client persistent state, so
whatever a client retains between rounds is attributable to the adapter and to
nothing else. A trainable `pre_classifier` would add 590,592 moving parameters
and a second channel a reviewer could point at.

Note on ``modules_to_save``: PEFT auto-appends the classification head for
``TaskType.SEQ_CLS`` (and speculatively "score", which DistilBERT does not have).
Passing our own list on top produced duplicated entries in the saved
`adapter_config.json`. ``_dedupe_modules_to_save`` normalises it after wrapping.
"""

from __future__ import annotations

from pathlib import Path

MODEL_NAME = "distilbert-base-uncased"

# LoRA recipe -- single source of truth for the federated phases too.
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_lin", "v_lin"]

#: Head modules PEFT keeps TRAINABLE and wraps via ``modules_to_save``.
#: `pre_classifier` appears here only when it is trainable. When frozen it is
#: loaded into the base model instead -- note PEFT still *writes* it into the
#: saved adapter (a SEQ_CLS adapter carries its head so it stays self-contained),
#: but the bytes are identical to the source checkpoint because nothing trains it.
LORA_MODULES_TO_SAVE = ["pre_classifier", "classifier"]
LORA_MODULES_TO_SAVE_FROZEN_PRE = ["classifier"]

#: Expected trainable counts. Pinned so a PEFT rename or a silently-unfrozen
#: module fails on run 1 rather than corrupting a sweep.
EXPECTED_TRAINABLE = 784_188  # freeze_pre_classifier=False
EXPECTED_TRAINABLE_FROZEN_PRE = 193_596  # freeze_pre_classifier=True
EXPECTED_LORA_PARAMS = 147_456

#: Adapter directory, the default source of trained head weights.
DEFAULT_HEAD_CHECKPOINT = "artifacts/baseline/distilbert_lora_r8"

_HEAD_TENSORS = (
    "pre_classifier.weight",
    "pre_classifier.bias",
)


def _dedupe_modules_to_save(model) -> list[str]:
    """Normalise ``peft_config.modules_to_save`` in place, preserving order.

    PEFT appends its own SEQ_CLS head names to whatever we pass, so the saved
    config could read ``["pre_classifier", "classifier", "classifier", "score"]``
    -- `classifier` twice and `score`, which does not exist on DistilBERT.
    Harmless at runtime, but it corrupts any key-ordering or arity assert built
    on top of the config, which the federated phases rely on.
    """
    module_names = [mod_name for mod_name, _ in model.named_modules()]
    clean: list[str] = []
    for cfg in model.peft_config.values():
        seen, clean = set(), []
        for name in list(cfg.modules_to_save or []):
            if name in seen:
                continue
            # keep only modules the model actually has
            if not any(name in mod_name for mod_name in module_names):
                continue
            seen.add(name)
            clean.append(name)
        cfg.modules_to_save = clean
    return clean


def _load_head_weights(base, checkpoint_dir: Path, tensors=_HEAD_TENSORS) -> None:
    """Copy trained head tensors from a saved PEFT adapter into ``base``.

    The saved adapter stores them as ``base_model.model.<name>``. Raises if a
    tensor is absent or mis-shaped -- a silently skipped load would leave random
    weights frozen in the forward path, which is the exact failure this function
    exists to prevent.
    """
    import torch
    from safetensors.torch import load_file

    path = Path(checkpoint_dir)
    if not path.is_absolute():
        from fedknob.data.massive import PROJECT_ROOT

        path = PROJECT_ROOT / path
    weights_file = path / "adapter_model.safetensors"
    if not weights_file.exists():
        raise FileNotFoundError(
            f"head checkpoint not found: {weights_file}\n"
            f"Run scripts/train_lora_massive.py first, or pass head_init_from=None "
            f"together with freeze_pre_classifier=False."
        )

    state = load_file(str(weights_file))
    loaded = []
    for name in tensors:
        key = f"base_model.model.{name}"
        if key not in state:
            raise KeyError(
                f"{key} missing from {weights_file}. Available head keys: "
                f"{sorted(k for k in state if 'classifier' in k)}"
            )
        module_path, attr = name.rsplit(".", 1)
        module = base.get_submodule(module_path)
        target = getattr(module, attr)
        src = state[key]
        if tuple(src.shape) != tuple(target.shape):
            raise ValueError(
                f"shape mismatch for {name}: checkpoint {tuple(src.shape)} vs "
                f"model {tuple(target.shape)}"
            )
        with torch.no_grad():
            target.copy_(src.to(target.dtype))
        loaded.append(name)
    if len(loaded) != len(tensors):
        raise RuntimeError(f"expected {len(tensors)} head tensors, loaded {loaded}")


def build_lora_model(
    num_labels: int,
    r: int = LORA_R,
    alpha: int = LORA_ALPHA,
    dropout: float = LORA_DROPOUT,
    freeze_pre_classifier: bool = False,
    head_init_from: str | Path | None = None,
    verify_trainable: bool = True,
):
    """Return DistilBERT-for-sequence-classification wrapped with LoRA (PEFT).

    The 66M-parameter backbone is always frozen.

    Parameters
    ----------
    freeze_pre_classifier:
        ``False`` (default) reproduces the centralized recipe: both head modules
        trainable, 784,188 trainable parameters.
        ``True`` is the federated recipe: `pre_classifier` is initialised from
        ``head_init_from`` and frozen, leaving 193,596 trainable parameters.
    head_init_from:
        Directory of a saved PEFT adapter to take `pre_classifier` weights from.
        Required when ``freeze_pre_classifier=True``; defaults to the saved
        adapter. Ignored otherwise.
    verify_trainable:
        Assert the trainable-parameter count matches the pinned constant. Leave
        on -- it is the cheapest guard against a silently-unfrozen module.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification

    base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    if freeze_pre_classifier:
        # Order matters: load trained weights, THEN freeze. Freezing the random
        # initialisation is the failure this whole branch exists to avoid.
        _load_head_weights(base, Path(head_init_from or DEFAULT_HEAD_CHECKPOINT))
        base.pre_classifier.requires_grad_(False)
        modules_to_save = LORA_MODULES_TO_SAVE_FROZEN_PRE
    else:
        modules_to_save = LORA_MODULES_TO_SAVE

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(LORA_TARGET_MODULES),
        # COPY, never the module-level list: LoraConfig appends its SEQ_CLS head
        # defaults *in place*, so passing the constant corrupts it for the rest of
        # the process. A fresh interpreter then shows it clean, which makes the
        # resulting mislabelled artifacts very hard to trace.
        modules_to_save=list(modules_to_save),
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)

    # PEFT re-wraps modules_to_save entries as trainable copies; anything left
    # out stays frozen. Re-assert the freeze in case wrapping restored the flag.
    if freeze_pre_classifier:
        for name, param in model.named_parameters():
            if "pre_classifier" in name:
                param.requires_grad_(False)

    _dedupe_modules_to_save(model)

    if verify_trainable:
        expected = EXPECTED_TRAINABLE_FROZEN_PRE if freeze_pre_classifier else EXPECTED_TRAINABLE
        actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if actual != expected:
            raise AssertionError(
                f"trainable parameter count {actual:,} != expected {expected:,} "
                f"(freeze_pre_classifier={freeze_pre_classifier}). A PEFT version "
                f"change or an unfrozen module will show up here first."
            )
    return model


def trainable_parameter_report(model) -> dict:
    """Break trainable params into LoRA-adapter-proper vs classification head.

    Returns a dict with raw counts and percentages; also handy for the model
    card. `pct_trainable` matches PEFT's print_trainable_parameters; the
    federated-feasibility claim rests on `pct_lora_only` (the bytes that
    actually persist per household).

    ``pre_classifier_frozen`` records which head regime the model was built in,
    so a model card cannot silently claim the wrong trainable count.
    """
    n_lora = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora_" in n)
    n_head = sum(
        p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n
    )
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = n_lora + n_head
    pre_frozen = not any(
        p.requires_grad for n, p in model.named_parameters() if "pre_classifier" in n
    )
    # Read the live config rather than a module constant -- see the copy note in
    # build_lora_model. This is what actually governs the saved adapter.
    mts: list[str] = []
    for cfg in getattr(model, "peft_config", {}).values():
        mts = list(cfg.modules_to_save or [])
        break
    return {
        "lora_adapter_params": n_lora,
        "classification_head_params": n_head,
        "trainable_params": n_trainable,
        "total_params": n_total,
        "pct_trainable": 100.0 * n_trainable / n_total,
        "pct_lora_only": 100.0 * n_lora / n_total,
        "pre_classifier_frozen": pre_frozen,
        "modules_to_save": mts,
    }
