"""Week 2 -- LoRA parameter-efficient baseline for FedKnob.

Swaps the Week 1 full fine-tune for a rank-8 LoRA baseline on MASSIVE (en):
same data pipeline, same tokenization (DistilBERT tokenizer, max_length=64),
same metrics -- but only the LoRA A/B matrices (+ classification head) train.
Goal: match the Week 1 accuracy (~85-88%) while the persistent per-household
artifact (the LoRA matrices) is <1% of the parameters.

Data is loaded via fedknob.data.massive -- the SAME single source-of-truth
loader the Week 1 baseline used -- so the two runs are directly comparable.
Tokenized datasets are cached to disk under artifacts/tokenized/ and reused.

Usage:
    python train_lora_massive.py --quick     # ~2k-example smoke run (1-2 min)
    python train_lora_massive.py             # full run (~5 min on an 8 GB GPU)

    # security-oriented diagnostics (same flags as Week 1):
    python train_lora_massive.py --per-class-report
    python train_lora_massive.py --asr-trigger "cf" --asr-target weather_query

What to expect:
    full : within ~1-2 pts of the Week 1 full-fine-tune accuracy on the
           MASSIVE (en) test split (Week 1 reference: ~82-85%).

Outputs (under artifacts/baseline/distilbert_lora_r8/):
    adapter_model.safetensors / adapter_config.json  -- PEFT adapter (gitignored)
    lora_metrics.json                                -- test metrics  (committed)
    MODEL_CARD.md                                    -- model card    (committed)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import transformers

# Shared helpers from the Week 1 baseline script (same directory): identical
# metrics and diagnostics, so Week 1 vs Week 2 numbers are apples-to-apples.
from train_baseline_massive import (
    attack_success_rate,
    compute_metrics,
    per_class_report,
)
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from fedknob.data.massive import PROJECT_ROOT, load_massive_en
from fedknob.models.distilbert_lora import (
    DEFAULT_HEAD_CHECKPOINT,
    EXPECTED_TRAINABLE,
    EXPECTED_TRAINABLE_FROZEN_PRE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_MODULES_TO_SAVE,
    LORA_MODULES_TO_SAVE_FROZEN_PRE,
    LORA_R,
    LORA_TARGET_MODULES,
    MODEL_NAME,
    build_lora_model,
    trainable_parameter_report,
)
from fedknob.utils.seeding import set_seed

transformers.logging.set_verbosity_error()

MAX_LENGTH = 64
TOKENIZED_CACHE = PROJECT_ROOT / "artifacts" / "tokenized" / "massive_en_distilbert_max64"
BASELINE_METRICS = PROJECT_ROOT / "artifacts" / "baseline" / "baseline_metrics.json"


def tokenize_with_cache(ds, tok, label2id, cache_dir: Path, no_cache: bool = False):
    """Tokenize the DatasetDict (max_length=64) and cache the result to disk.

    The cache key is the directory name (tokenizer + max_length are fixed for
    the project). A label_map.json saved next to the arrow files guards against
    silently reusing a cache built with a different label mapping.
    """
    from datasets import load_from_disk

    label_map_path = cache_dir / "label_map.json"
    if not no_cache and cache_dir.exists() and label_map_path.exists():
        with open(label_map_path) as f:
            cached_map = json.load(f)
        if cached_map == label2id:
            print(f"  loading tokenized dataset from cache: {cache_dir}")
            return load_from_disk(str(cache_dir))
        print("  cached label map differs -- re-tokenizing")

    def preprocess(batch):
        enc = tok(batch["text"], truncation=True, max_length=MAX_LENGTH)
        enc["labels"] = [label2id[lab] for lab in batch["label"]]
        return enc

    cols = ds["train"].column_names
    ds = ds.map(preprocess, batched=True, remove_columns=cols)

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(cache_dir))
    with open(label_map_path, "w") as f:
        json.dump(label2id, f, indent=2)
    print(f"  tokenized dataset cached to: {cache_dir}")
    return ds


def compare_with_week1(lora_acc: float, train_split: str = "train") -> None:
    """Print the Week 2 exit-criteria check against the Week 1 metrics file.

    The gate only means something when both sides fitted the same data. A proxy
    init fitted on the 2,033-row validation split is *expected* to fall well
    short of a full fine-tune on 11,514 rows, so comparing them would report a
    correct run as a failure.
    """
    if train_split != "train":
        print("\n=== Week 2 exit criteria: SKIPPED ===")
        print(
            f"  Fitted on '{train_split}', not 'train'. The Week 1 comparison "
            f"assumes the same fitting data;\n  a lower accuracy here is the "
            f"expected consequence of a smaller proxy set, not a failure."
        )
        return
    if not BASELINE_METRICS.exists():
        print(
            "\n[exit criteria] Week 1 baseline_metrics.json not found -- "
            "run scripts/train_baseline_massive.py first to compare."
        )
        return
    with open(BASELINE_METRICS) as f:
        week1 = json.load(f)
    full_acc = week1.get("eval_accuracy")
    if full_acc is None:
        print("\n[exit criteria] eval_accuracy missing from Week 1 metrics.")
        return
    gap = (full_acc - lora_acc) * 100
    print("\n=== Week 2 exit criteria: LoRA vs Week 1 full fine-tune ===")
    print(f"  Week 1 full fine-tune accuracy : {full_acc:.4f}")
    print(f"  Week 2 LoRA (r8) accuracy      : {lora_acc:.4f}")
    print(f"  gap                            : {gap:+.2f} pts")
    if gap <= 2.0:
        print("  PASS -- LoRA baseline is within ~1-2 pts of the full fine-tune.")
    else:
        print(
            "  NOT YET -- gap exceeds 2 pts; try more epochs (--epochs 5), or"
            " verify lr=2e-4 and that the classification head is trainable."
        )


def write_model_card(out_dir: str, metrics: dict, report: dict, args) -> None:
    """Write a model card next to the adapter (committed; weights gitignored)."""
    acc = metrics.get("eval_accuracy", float("nan"))
    lora_n = f"{report['lora_adapter_params']:,}"
    head_n = f"{report['classification_head_params']:,}"
    train_n = f"{report['trainable_params']:,}"
    total_n = f"{report['total_params']:,}"
    pct_lora = f"{report['pct_lora_only']:.2f}"
    pct_train = f"{report['pct_trainable']:.2f}"
    # Report what this run actually did, not the module-level default -- a card that
    # names the wrong head regime is worse than one that omits it.
    frozen = report.get("pre_classifier_frozen", False)
    # Prefer the model's live config over a module constant -- PEFT mutates the list
    # it is handed, so the constant may have been appended to during this process.
    mts = report.get("modules_to_save") or (
        LORA_MODULES_TO_SAVE_FROZEN_PRE if frozen else LORA_MODULES_TO_SAVE
    )
    head_src = getattr(args, "head_init_from", None) or DEFAULT_HEAD_CHECKPOINT
    regime = (
        f"`pre_classifier` frozen at trained values (from "
        f"`{head_src}`); trains LoRA + `classifier` only"
        if frozen
        else "`pre_classifier` + `classifier` both trainable"
    )
    # Mark whichever row of the head-regime table this card is actually describing.
    mark_central, mark_fed = ("", " **(this card)**") if frozen else (" **(this card)**", "")
    # Built here rather than inline: a markdown row must not wrap in the output, so
    # assembling it separately keeps the source inside the line-length limit.
    fed_row = (
        f"| federated{mark_fed} | frozen, loaded from "
        f"`{DEFAULT_HEAD_CHECKPOINT}` | trainable, aggregated | "
        f"{EXPECTED_TRAINABLE_FROZEN_PRE:,} |"
    )
    split = getattr(args, "train_split", "train")
    proxy = split != "train"
    title = (
        "FedKnob server-side proxy init"
        if proxy
        else "FedKnob LoRA baseline, frozen-head regime"
        if frozen
        else "FedKnob Week 2 LoRA baseline"
    )
    summary = (
        f"""Rank-{args.r} LoRA fine-tune of `{MODEL_NAME}` for 60-way intent classification
on Amazon MASSIVE (en), trained in the head regime the federated phases use:
`pre_classifier` is loaded from `{head_src}` and held fixed, so only the LoRA
matrices and the 60-way `classifier` move. It is the control run that measures
what that freeze costs against the centralised Week 2 baseline."""
        if frozen
        else f"""Rank-{args.r} LoRA fine-tune of `{MODEL_NAME}` for 60-way intent classification
on Amazon MASSIVE (en). Parameter-efficient counterpart of the Week 1 full
fine-tune; the persistent per-household artifact in the federated phases is the
LoRA adapter saved here."""
    )
    reproduce = (
        f"""python scripts/train_lora_massive.py --train-split {split} \\
    --out {out_dir.replace(os.sep, "/")}"""
        if proxy
        else f"""python scripts/train_lora_massive.py --freeze-pre-classifier \\
    --out {out_dir.replace(os.sep, "/")}"""
        if frozen
        else "python scripts/train_lora_massive.py"
    )
    if proxy:
        summary = f"""Rank-{args.r} LoRA fine-tune of `{MODEL_NAME}` for 60-way intent
classification on Amazon MASSIVE (en), fitted on the **{split}** split only.

This is the server-side initialisation for the federated phases. The federated
clients are built from the *training* split, so an adapter fitted on that split
has already seen every client's private data and cannot be a legitimate FL
starting point -- the server would have had to centralise the very data
federation exists to keep local. Fitting on a small held-out split instead makes
the round-0 model something a server could genuinely hold. Expect materially
lower standalone accuracy than the Week 2 baseline: that gap is the headroom the
federated rounds are supposed to close, and closing it is the result."""
    fit_note = (
        "server-side proxy init, disjoint from the client partitions"
        if proxy
        else "the split the federated clients are partitioned from"
    )
    card = f"""# Model Card -- {title} (DistilBERT / MASSIVE)

## Summary

{summary}

- **Fitted on:** MASSIVE en-US `{split}` split -- {fit_note}
- **Evaluated on:** MASSIVE en-US `test` split (never fitted on by any phase)

- **Base model:** {MODEL_NAME} (66M params, frozen)
- **Task:** sequence classification (60 MASSIVE intents)
- **Method:** PEFT LoRA -- r={args.r}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT},
  target_modules={LORA_TARGET_MODULES}, modules_to_save={mts}
- **Head regime:** {regime}
- **Dataset:** Amazon MASSIVE en-US via `fedknob.data.massive`
  (same loader as Week 1; official alexa/massive 1.0 release)
- **Tokenization:** DistilBERT tokenizer, max_length={MAX_LENGTH} (cached to disk)
- **License:** Apache-2.0

## Parameter efficiency

| component | params | % of total |
|---|---|---|
| LoRA A/B matrices (persistent per-household adapter) | {lora_n} | {pct_lora}% |
| classification head (modules_to_save; shared, new init) | {head_n} | -- |
| total trainable | {train_n} | {pct_train}% |
| total | {total_n} | 100% |

The <1% feasibility claim refers to the LoRA matrices proper -- the bytes that
persist on-device per household. The classification head belongs to the shared
global model, not the per-household adapter.

### Head regimes

`build_lora_model()` supports two configurations of the classification head.

| regime | `pre_classifier` | `classifier` | trainable |
|---|---|---|---|
| centralised{mark_central} | trainable | trainable | {EXPECTED_TRAINABLE:,} |
{fed_row}

`pre_classifier` is created by `from_pretrained(num_labels=...)` with random
weights -- `distilbert-base-uncased` ships no classification head -- so the
centralised run trains it and the federated phases reuse it frozen. That leaves
LoRA `B` (73,728 params) as the only per-client persistent state, which keeps the
adapter the sole carrier channel for anything a client retains between rounds.

Holding `pre_classifier` fixed costs **0.17 accuracy points** (0.8806 -> 0.8790)
while freezing 590,592 parameters, 75% of the trainable budget. At n=2,974 the
accuracy standard error is 0.59 points, so that gap is **0.28 SE** --
indistinguishable from zero, and the sign is inconsistent across metrics.
Side-by-side figures in `artifacts/baseline/head_regime_control.json`.

## Training

- Learning rate: {args.lr} (LoRA; 10x the full-fine-tune lr)
- Epochs: {args.epochs}, batch size {args.batch_size}, max sequence length {MAX_LENGTH}
- Optimizer/schedule: HF `Trainer` defaults (AdamW); fp16 when CUDA available
- Seed: {args.seed}

Reproduce with:

```bash
{reproduce}
```

## Evaluation (MASSIVE test split)

- accuracy: {acc:.4f}
- full metrics in `lora_metrics.json` next to this card

Exit criteria (Week 2): within ~1-2 pts of the Week 1 full-fine-tune accuracy
(`artifacts/baseline/baseline_metrics.json`).

## Intended use & limitations

Research baseline only -- the clean (un-poisoned) LoRA reference for the
FedKnob backdoor-contagion experiments. Not for production use.
"""
    with open(os.path.join(out_dir, "MODEL_CARD.md"), "w") as f:
        f.write(card)


def main() -> None:
    ap = argparse.ArgumentParser(description="Week 2: rank-8 LoRA baseline on MASSIVE (en-US)")
    ap.add_argument("--quick", action="store_true", help="small subset smoke run")
    ap.add_argument(
        "--epochs",
        type=float,
        default=10.0,
        help="3-5 epochs; LoRA usually needs more than full FT",
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4, help="LoRA LR (10x the full-fine-tune 2e-5)")
    ap.add_argument("--r", type=int, default=LORA_R, help="LoRA rank")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no-cache", action="store_true", help="ignore the on-disk tokenized cache and re-tokenize"
    )
    ap.add_argument(
        "--train-split",
        choices=["train", "validation"],
        default="train",
        help="MASSIVE split to FIT on. 'train' (default) is the Week 2 "
        "baseline. 'validation' (2,033 utterances) produces the "
        "server-side proxy init for the federated phases: the "
        "training split is partitioned into clients, so an adapter "
        "fitted on it has already seen every client's private rows "
        "and cannot serve as an FL starting point. Evaluation is on "
        "test either way.",
    )
    ap.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "artifacts" / "baseline" / "distilbert_lora_r8"),
        help="output dir (default: artifacts/baseline/distilbert_lora_r8)",
    )
    ap.add_argument(
        "--freeze-pre-classifier",
        action="store_true",
        help="freeze pre_classifier at its trained values and train only "
        "LoRA + classifier (193,596 params). Weights come from "
        "--head-init-from. This is the federated-phase recipe; use it "
        "to measure what freezing costs against the 784,188-param run.",
    )
    ap.add_argument(
        "--head-init-from",
        default=DEFAULT_HEAD_CHECKPOINT,
        help="adapter dir supplying the trained pre_classifier weights "
        "when --freeze-pre-classifier is set "
        f"(default: {DEFAULT_HEAD_CHECKPOINT})",
    )
    ap.add_argument(
        "--per-class-report",
        action="store_true",
        help="save per-class P/R/F1 + confusion matrix after training",
    )
    ap.add_argument(
        "--asr-trigger",
        default=None,
        help="trigger phrase to measure a baseline (clean) ASR control",
    )
    ap.add_argument(
        "--asr-target", default=None, help="target intent NAME (or id) the trigger should force"
    )
    args = ap.parse_args()

    set_seed(args.seed)

    print("Loading Amazon MASSIVE (en-US) ...")
    ds = load_massive_en("en-US")

    # Same stable name->id map as Week 1 (sorted intent names).
    label_names = sorted({lab for split in ds for lab in ds[split]["label"]})
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for name, i in label2id.items()}
    num_labels = len(label_names)
    print(f"  intents (labels): {num_labels}")
    print(f"  train/val/test  : {len(ds['train'])}/{len(ds['validation'])}/{len(ds['test'])}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    # keep raw test utterances for the optional ASR probe before columns drop.
    raw_test_texts = list(ds["test"]["text"])

    ds = tokenize_with_cache(ds, tok, label2id, TOKENIZED_CACHE, no_cache=args.no_cache)

    # Fit on --train-split, always evaluate on test. The two are disjoint in
    # every case, and test is never fitted on by any phase of this project.
    train_ds, eval_ds = ds[args.train_split], ds["test"]
    if args.train_split != "train":
        print(
            f"  fitting on      : {args.train_split} split ({len(train_ds)} rows) "
            f"-- SERVER-SIDE PROXY INIT, disjoint from the client partitions"
        )
    if args.quick:
        train_ds = train_ds.select(range(min(2000, len(train_ds))))
        eval_ds = eval_ds.select(range(min(1000, len(eval_ds))))
        raw_test_texts = raw_test_texts[: len(eval_ds)]
        print("  [quick mode] using a small subset for a fast smoke test")

    if args.freeze_pre_classifier:
        # The frozen run READS pre_classifier from --head-init-from and, left at the
        # default --out, would write its own adapter back over that same directory --
        # silently making the next run initialise from its own previous output.
        src = (
            Path(args.head_init_from)
            if Path(args.head_init_from).is_absolute()
            else PROJECT_ROOT / args.head_init_from
        ).resolve()
        if Path(args.out).resolve() == src:
            raise SystemExit(
                f"--out and --head-init-from are the same directory:\n  {src}\n"
                f"The frozen run would overwrite the checkpoint it initialises from. "
                f"Pass a different --out, e.g.\n"
                f"  --out artifacts/baseline/distilbert_lora_r8_frozenpre"
            )
        print(
            f"  head regime     : pre_classifier FROZEN "
            f"({EXPECTED_TRAINABLE_FROZEN_PRE:,} trainable)"
        )
        print(f"  head weights    : {src}")
    else:
        print(f"  head regime     : pre_classifier trainable ({EXPECTED_TRAINABLE:,} trainable)")

    model = build_lora_model(
        num_labels,
        r=args.r,
        freeze_pre_classifier=args.freeze_pre_classifier,
        head_init_from=args.head_init_from if args.freeze_pre_classifier else None,
    )

    # Week 2 gate: confirm the parameter-efficiency property explicitly.
    model.print_trainable_parameters()
    report = trainable_parameter_report(model)
    print(
        f"  LoRA adapter only      : {report['lora_adapter_params']:,} "
        f"({report['pct_lora_only']:.2f}% of total) <- persistent per-household bytes"
    )
    print(
        f"  classification head    : {report['classification_head_params']:,} "
        f"(newly initialised; shared global, not part of the adapter)"
    )
    assert report["pct_lora_only"] < 1.0, "LoRA adapter must be <1% of parameters"

    use_fp16 = torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        fp16=use_fp16,
        label_names=["labels"],  # PEFT wrapping hides them from Trainer inference
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    print("\n=== LoRA baseline metrics (test) ===")
    for key in (
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
        "precision_macro",
        "recall_macro",
        "top5_accuracy",
    ):
        ek = f"eval_{key}"
        if ek in metrics:
            print(f"  {key:<18}: {metrics[ek]:.4f}")

    compare_with_week1(metrics.get("eval_accuracy", 0.0), args.train_split)

    os.makedirs(args.out, exist_ok=True)
    eval_metrics = {k: v for k, v in metrics.items() if k.startswith("eval_")}
    eval_metrics["trainable_parameter_report"] = report
    # Provenance travels with the checkpoint. ``saw_client_rows`` is the whole
    # question in one boolean: the federated clients are built from the training
    # split, so an adapter fitted on it has already seen their private data and
    # cannot be a legitimate FL starting point.
    eval_metrics["provenance"] = {
        "train_split": args.train_split,
        "n_train": len(train_ds),
        "eval_split": "test",
        "n_eval": len(eval_ds),
        "saw_client_rows": args.train_split == "train",
        "freeze_pre_classifier": bool(args.freeze_pre_classifier),
        "head_init_from": (args.head_init_from if args.freeze_pre_classifier else None),
        "epochs": args.epochs,
        "lr": args.lr,
        "r": args.r,
        "seed": args.seed,
    }
    with open(os.path.join(args.out, "lora_metrics.json"), "w") as f:
        json.dump(eval_metrics, f, indent=2)

    # --- optional diagnostics (same as Week 1) -------------------------------
    if args.per_class_report:
        per_class_report(trainer, eval_ds, id2label, args.out)

    if args.asr_trigger is not None and args.asr_target is not None:
        if args.asr_target in label2id:
            target_id = label2id[args.asr_target]
        else:
            target_id = int(args.asr_target)
        attack_success_rate(trainer, tok, raw_test_texts, target_id, args.asr_trigger, num_labels)

    # save_pretrained on a PEFT model writes ONLY the adapter
    # (adapter_model.safetensors + adapter_config.json), not the 66M backbone.
    model.save_pretrained(args.out)
    write_model_card(args.out, metrics, report, args)
    print(f"\nSaved LoRA adapter + model card to: {args.out}")
    print("  reload with: PeftModel.from_pretrained(build_model(60), <out_dir>)")


if __name__ == "__main__":
    main()
