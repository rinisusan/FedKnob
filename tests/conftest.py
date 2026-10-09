"""Shared fixtures.

Some slow tests build a real model in the federated head regime, which reads
``pre_classifier`` from ``artifacts/baseline/distilbert_lora_r8/``. Those adapter
weights are deliberately **not** committed -- ``.gitignore`` keeps the weights out
and the metrics and model card in -- so the file exists only after the LoRA run
has been done on that machine.

On a fresh clone or a CI runner it is absent, and the tests fail with a
``FileNotFoundError`` that reads like a broken build rather than a missing
prerequisite. They skip instead, naming the command that produces the file.

The distinction matters: a failure means the code is wrong, a skip means the
artifact is not here yet. Conflating the two trains people to ignore red CI.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HEAD_CHECKPOINT = (
    REPO_ROOT / "artifacts" / "baseline" / "distilbert_lora_r8" / "adapter_model.safetensors"
)


@pytest.fixture
def head_checkpoint() -> Path:
    """Skip unless the trained adapter is on disk.

    Request this in any test that calls ``build_lora_model(...,
    freeze_pre_classifier=True)`` -- that path loads the frozen head from the
    checkpoint and cannot run without it.
    """
    if not HEAD_CHECKPOINT.exists():
        pytest.skip(
            f"{HEAD_CHECKPOINT.relative_to(REPO_ROOT)} not present. Adapter "
            f"weights are gitignored, so this test runs only after: "
            f"python scripts/train_lora_massive.py"
        )
    return HEAD_CHECKPOINT
