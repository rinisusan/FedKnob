"""Week 2 LoRA recipe tests: config constants + (if peft installed) wiring.

Covers both head regimes:
    freeze_pre_classifier=False -> 784,188 trainable (Week 1/2 baseline)
    freeze_pre_classifier=True  -> 193,596 trainable (federated phases)
"""

import json
import struct

import pytest

from fedknob.data.massive import PROJECT_ROOT
from fedknob.models import distilbert_lora as dl

# --------------------------------------------------------------------------
# constants -- no torch/peft needed
# --------------------------------------------------------------------------


def test_week2_recipe_constants():
    """The proposal's Week 2 recipe is pinned: r=8, alpha=16, dropout=0.05."""
    assert dl.LORA_R == 8
    assert dl.LORA_ALPHA == 16
    assert dl.LORA_DROPOUT == 0.05
    assert dl.LORA_TARGET_MODULES == ["q_lin", "v_lin"]
    assert set(dl.LORA_MODULES_TO_SAVE) == {"pre_classifier", "classifier"}
    assert dl.LORA_MODULES_TO_SAVE_FROZEN_PRE == ["classifier"]
    assert dl.MODEL_NAME == "distilbert-base-uncased"


def test_expected_trainable_constants_are_arithmetically_right():
    """Recompute the pinned counts from first principles.

    If a constant is edited without the arithmetic behind it, this fails.
    """
    hidden, layers, rank, labels = 768, 6, 8, 60
    lora = 2 * layers * (rank * hidden + hidden * rank)  # q_lin,v_lin x A,B
    pre = hidden * hidden + hidden
    cls = hidden * labels + labels

    assert lora == dl.EXPECTED_LORA_PARAMS == 147_456
    assert lora + pre + cls == dl.EXPECTED_TRAINABLE == 784_188
    assert lora + cls == dl.EXPECTED_TRAINABLE_FROZEN_PRE == 193_596
    # freezing pre_classifier removes exactly its own parameter count
    assert dl.EXPECTED_TRAINABLE - dl.EXPECTED_TRAINABLE_FROZEN_PRE == pre == 590_592


def test_head_checkpoint_contains_the_tensors_we_load():
    """The Week-2 adapter must hold the pre_classifier weights, under the exact
    keys ``_load_head_weights`` requests.

    Reads the safetensors header directly (8-byte little-endian length prefix
    then JSON) so the test needs neither torch nor safetensors installed.
    """
    ckpt = PROJECT_ROOT / dl.DEFAULT_HEAD_CHECKPOINT / "adapter_model.safetensors"
    if not ckpt.exists():
        pytest.skip(f"Week-2 adapter not present at {ckpt}")

    with open(ckpt, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))

    expected_shapes = {
        "pre_classifier.weight": [768, 768],
        "pre_classifier.bias": [768],
    }
    for name in dl._HEAD_TENSORS:
        key = f"base_model.model.{name}"
        assert key in header, f"{key} missing from {ckpt}"
        assert header[key]["shape"] == expected_shapes[name]


# --------------------------------------------------------------------------
# wiring -- needs peft + torch + the base checkpoint
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_lora_model_builds_and_is_param_efficient():
    """Week 1/2 regime: both head modules trainable."""
    pytest.importorskip("peft")
    pytest.importorskip("torch")
    model = dl.build_lora_model(num_labels=60)
    report = dl.trainable_parameter_report(model)
    assert report["lora_adapter_params"] == dl.EXPECTED_LORA_PARAMS
    assert report["pct_lora_only"] < 1.0
    assert report["classification_head_params"] > 0
    assert report["trainable_params"] == dl.EXPECTED_TRAINABLE
    assert report["pre_classifier_frozen"] is False


@pytest.mark.slow
def test_frozen_pre_classifier_regime():
    """Federated regime: pre_classifier loaded from Week 2, then frozen.

    193,596 trainable = LoRA A+B (147,456) + classifier (46,140). This is the
    count Week 4's no-op-training guard asserts against.
    """
    pytest.importorskip("peft")
    pytest.importorskip("torch")
    ckpt = PROJECT_ROOT / dl.DEFAULT_HEAD_CHECKPOINT / "adapter_model.safetensors"
    if not ckpt.exists():
        pytest.skip("Week-2 adapter needed to initialise the frozen head")

    model = dl.build_lora_model(num_labels=60, freeze_pre_classifier=True)
    report = dl.trainable_parameter_report(model)

    assert report["trainable_params"] == dl.EXPECTED_TRAINABLE_FROZEN_PRE
    assert report["lora_adapter_params"] == dl.EXPECTED_LORA_PARAMS
    assert report["classification_head_params"] == 46_140  # classifier only
    assert report["pre_classifier_frozen"] is True

    # no pre_classifier parameter may carry a gradient
    assert not any(p.requires_grad for n, p in model.named_parameters() if "pre_classifier" in n)


@pytest.mark.slow
def test_frozen_head_is_trained_not_random():
    """The frozen pre_classifier must equal the Week-2 checkpoint.

    Freezing the *random* initialisation is the failure this whole code path
    exists to prevent, and it is silent -- the model still trains and still
    produces a plausible loss curve.
    """
    pytest.importorskip("peft")
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    ckpt = PROJECT_ROOT / dl.DEFAULT_HEAD_CHECKPOINT / "adapter_model.safetensors"
    if not ckpt.exists():
        pytest.skip("Week-2 adapter needed")

    state = safetensors.load_file(str(ckpt))
    model = dl.build_lora_model(num_labels=60, freeze_pre_classifier=True)

    for name in dl._HEAD_TENSORS:
        want = state[f"base_model.model.{name}"]
        module_path, attr = name.rsplit(".", 1)
        got = getattr(model.base_model.model.get_submodule(module_path), attr)
        assert torch.allclose(got.cpu(), want.cpu(), atol=1e-6), (
            f"{name} does not match the Week-2 checkpoint -- the load silently "
            f"failed and random weights are frozen in the forward path"
        )


@pytest.mark.slow
def test_modules_to_save_has_no_duplicates_or_phantoms():
    """adapter_config.json must not record duplicated or non-existent modules.

    PEFT auto-appends SEQ_CLS head names to whatever we pass, which previously
    produced ["pre_classifier", "classifier", "classifier", "score"] -- with
    `classifier` twice and `score`, which DistilBERT does not have.
    """
    pytest.importorskip("peft")
    pytest.importorskip("torch")
    for frozen in (False, True):
        ckpt = PROJECT_ROOT / dl.DEFAULT_HEAD_CHECKPOINT / "adapter_model.safetensors"
        if frozen and not ckpt.exists():
            continue
        model = dl.build_lora_model(num_labels=60, freeze_pre_classifier=frozen)
        for cfg in model.peft_config.values():
            mts = list(cfg.modules_to_save)
            assert len(mts) == len(set(mts)), f"duplicates in {mts}"
            assert "score" not in mts, f"phantom 'score' module in {mts}"
            if frozen:
                assert "pre_classifier" not in mts, (
                    "a frozen pre_classifier must not sit in modules_to_save -- "
                    "that wrapper exists to make a module trainable"
                )


@pytest.mark.slow
def test_trainable_assert_fires_on_mismatch(monkeypatch):
    """verify_trainable must actually raise, not just compute."""
    pytest.importorskip("peft")
    pytest.importorskip("torch")
    monkeypatch.setattr(dl, "EXPECTED_TRAINABLE", 1)
    with pytest.raises(AssertionError, match="trainable parameter count"):
        dl.build_lora_model(num_labels=60)
