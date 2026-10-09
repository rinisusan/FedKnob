"""Round-0 gate -- run this before any federated code, and before every run.

No Flower, no Ray, no clients, no rounds. It answers one question:

    if we warm-start the model and round-trip its parameters through the wire
    format, do we still have the Week-2 model?

If the answer is no, nothing downstream is worth debugging, because every
federated result is built on this exact path. The check costs about a minute.

    python scripts/fl_round0_gate.py                    # warm start (default)
    python scripts/fl_round0_gate.py --init random      # cold start
    python scripts/fl_round0_gate.py --mode split_ab    # 14 keys instead of 26

Four assertions:

  1. trainable parameter count      == 193,596
  2. transmit key count             == 26 (ephemeral) / 14 (split_ab)
  3. warm-start accuracy            ~= 0.8806   (the Week-2 number)
  4. accuracy after get/set params  == exactly the above

(4) is the one that matters most. Flower ships a bare list of arrays with no
names, so position is the whole contract; a get/set ordering mismatch corrupts
weights without raising, and every count and shape check still passes. Only
comparing accuracy before and after a round-trip catches it.

Assertion (3) allows a small tolerance rather than demanding 0.8806321...: the
Week-2 number was measured under fp16 autocast and this evaluates in fp32, so a
borderline utterance may flip. One example is 0.034 pts at n=2,974. A gap of more
than a few examples is a real problem; one or two is precision. (In practice the
counts match exactly -- 2619/2974 either way.)

A NOTE ON "COLD START"
----------------------
``--init random`` is not a from-scratch model, and cannot be. ``pre_classifier``
is frozen in the federated regime, and freezing a *random* 768x768 projection
destroys accuracy -- that is the Week-3 head-regime result -- so it is always
loaded trained. A cold start therefore randomises only ``classifier`` (60-way)
and leaves LoRA ``B`` at PEFT's zero init.

The resulting model is a **random linear readout over a trained representation**,
which scores far above chance: 0.2942 at seed 42, against 1/60 = 0.0167. That
number is itself informative -- it is how much of the task the frozen
representation already solves before the classifier learns anything.

So the cold arm measures "can federation learn the readout and the adapters",
not "can federation learn the task from nothing". Worth stating plainly in the
write-up, because a reader will assume the latter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedknob.data.massive import PROJECT_ROOT  # noqa: E402
from fedknob.fl import data as D  # noqa: E402
from fedknob.fl import params as P  # noqa: E402
from fedknob.fl import task as T  # noqa: E402
from fedknob.models.distilbert_lora import (  # noqa: E402
    EXPECTED_TRAINABLE_FROZEN_PRE,
    build_lora_model,
)
from fedknob.utils.seeding import set_seed  # noqa: E402

WEEK2_ACCURACY = 0.8806
TOLERANCE_EXAMPLES = 3  # ~0.10 pts at n=2,974 -- fp16/fp32 slack

#: Upper bound for a cold start. NOT near chance (1/60), and that is expected:
#: a "cold" start in this architecture can only randomise `classifier`, because
#: `pre_classifier` is frozen and must therefore be loaded trained -- freezing a
#: random 768x768 projection destroys accuracy (the Week-3 head-regime result).
#: So a cold model is a *random 60-way readout over a trained representation*,
#: which lands well above chance: measured 0.2942 at seed 42. The assertion that
#: matters is that it is nowhere near the warm-start 0.8806, which would mean
#: warm_start() ran when it should not have.
COLD_START_CEILING = 0.70

#: Lower edge of the acceptable band for the proxy arm; the upper edge is the
#: Week 2 accuracy. Wide on purpose -- this catches "the checkpoint is missing or
#: is secretly the training-split one", not a few points of tuning.
PROXY_ACCURACY_BAND = (0.40, WEEK2_ACCURACY)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--init",
        choices=sorted(P.INIT_ARMS),
        default="week2",
        help="week2 = warm start from the training-split adapter "
        "(default); proxy = warm start from the held-out "
        "validation-split adapter; random = cold start",
    )
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="override the arm's checkpoint (supplies both the frozen "
        "pre_classifier and the warm-started adapter)",
    )
    ap.add_argument("--mode", choices=list(P.MODES), default=P.MODE_EPHEMERAL)
    ap.add_argument("--seed", type=int, default=42)
    # Default includes init and mode: the three arms are separate results, and a
    # fixed filename silently keeps only whichever ran last.
    ap.add_argument(
        "--out", default=None, help="default: artifacts/fl/round0_gate_<init>_<mode>.json"
    )
    args = ap.parse_args()
    if args.out is None:
        args.out = f"artifacts/fl/round0_gate_{args.init}_{args.mode}.json"

    failures: list[str] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        if not ok:
            failures.append(label)
        print(f"  {'PASS' if ok else '**FAIL**':<10} {label}{'   ' + detail if detail else ''}")

    # Seed BEFORE build: every client's LoRA A must start identical, or early
    # averaging is destructive for a reason unrelated to federation.
    set_seed(args.seed)
    print(f"\nBuilding model (init={args.init}, mode={args.mode}, seed={args.seed})")
    default_ckpt, warm = P.INIT_ARMS[args.init]
    ckpt = args.checkpoint or default_ckpt
    model = build_lora_model(num_labels=60, freeze_pre_classifier=True, head_init_from=ckpt)
    print(f"  pre_classifier (frozen, 590,592 params) from {ckpt}")
    if warm:
        n = P.warm_start(model, ckpt)
        print(f"  warm start: {n} tensors loaded from {ckpt}")

    print("\n1-2. STRUCTURE")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    check(trainable == EXPECTED_TRAINABLE_FROZEN_PRE, "trainable parameter count", f"{trainable:,}")
    keys = P.transmit_keys(model, args.mode)
    check(
        len(keys) == P.EXPECTED_KEYS[args.mode],
        "transmit key count",
        f"{len(keys)} (mode={args.mode})",
    )
    summary = P.params_summary(model, args.mode)
    print(
        f"             transmits {summary['transmit_kb_fp32']} KB/client/round; "
        f"{summary['local_params']:,} params stay local"
    )

    print("\n3. ACCURACY BEFORE ROUND-TRIP")
    test = D.load_central_eval("test")
    before = T.evaluate(model, test)
    acc_b, n = before["accuracy"], before["n_examples"]
    print(f"             {before['n_correct']}/{n} correct")
    if args.init == "week2":
        gap_examples = abs(acc_b - WEEK2_ACCURACY) * n
        check(
            gap_examples <= TOLERANCE_EXAMPLES,
            f"warm-start accuracy ~= {WEEK2_ACCURACY}",
            f"{acc_b:.4f}  ({gap_examples:.1f} examples from Week 2)",
        )
    elif args.init == "proxy":
        # A band, not a point. The proxy adapter is genuinely trained, so it must
        # clear the cold arm by a wide margin; but it saw 2,033 rows against Week
        # 2's 11,514, so landing at or above the Week 2 number would mean the
        # checkpoint is not what it claims to be -- most likely a proxy directory
        # that was actually fitted on the training split.
        lo, hi = PROXY_ACCURACY_BAND[0], WEEK2_ACCURACY
        check(
            lo < acc_b < hi,
            f"proxy init inside ({lo}, {hi})",
            f"{acc_b:.4f}  (fitted on held-out data; below Week 2 by design -- "
            f"that gap is the headroom the rounds must close)",
        )
    else:
        check(
            acc_b < COLD_START_CEILING,
            "cold start is untrained",
            f"{acc_b:.4f}  (random readout over a trained representation; "
            f"chance = {1 / 60:.4f}, warm = {WEEK2_ACCURACY})",
        )

    print("\n4. ACCURACY AFTER get_params -> set_params")
    P.set_params(model, P.get_params(model, args.mode), args.mode)
    after = T.evaluate(model, test)
    check(
        after["n_correct"] == before["n_correct"],
        "round-trip is the identity",
        f"{after['n_correct']}/{n} correct  (was {before['n_correct']})",
    )

    out = {
        "init": args.init,
        "mode": args.mode,
        "seed": args.seed,
        "checkpoint": ckpt,
        "warm_started": warm,
        "trainable_params": trainable,
        "transmit_keys": len(keys),
        "params_summary": summary,
        "accuracy_before": acc_b,
        "accuracy_after": after["accuracy"],
        "n_correct_before": before["n_correct"],
        "n_correct_after": after["n_correct"],
        "n_test": n,
        "week2_reference": WEEK2_ACCURACY,
        "passed": not failures,
    }
    path = Path(args.out)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 66)
    if failures:
        print(f"  {len(failures)} FAILURE(S): {', '.join(failures)}")
        print("  Stop here. Everything federated is built on this path.")
        raise SystemExit(1)
    print("  Round-0 gate PASSED -- the wire format preserves the model.")
    print(f"  Written to {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
