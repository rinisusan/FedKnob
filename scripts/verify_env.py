"""Verify the FedKnob toolchain is correctly installed.

Run this AFTER installing the requirements and BEFORE training:
    python verify_env.py

It prints package versions, whether CUDA is visible, your GPU + VRAM, and
whether bitsandbytes (QLoRA) is available (optional; used later).
"""
import importlib
import sys

CORE_PKGS = ["torch", "transformers", "peft", "datasets", "accelerate", "numpy"]


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    ok = True
    for name in CORE_PKGS:
        try:
            m = importlib.import_module(name)
            print(f"  [ok]      {name:14s} {getattr(m, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  [MISSING] {name:14s} {e}")

    try:
        import torch

        print(f"\nCUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  device : {torch.cuda.get_device_name(0)}")
            free, total = torch.cuda.mem_get_info()
            print(f"  VRAM   : {total / 1e9:.1f} GB total, {free / 1e9:.1f} GB free")
    except Exception as e:  # noqa: BLE001
        print(f"torch CUDA check failed: {e}")

    # Optional: bitsandbytes 4-bit (QLoRA). Not needed for the baseline.
    try:
        import bitsandbytes as bnb

        print(f"\nbitsandbytes: {bnb.__version__} (QLoRA available)")
    except Exception:  # noqa: BLE001
        print("\nbitsandbytes: not installed (OK for the baseline; needed for QLoRA later)")

    print("\nAll core packages present." if ok else "\nSome packages missing - see above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
