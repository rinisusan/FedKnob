"""Baseline trainer for FedKnob.

Full fine-tune of DistilBERT (~66M params, NO LoRA) on the Amazon MASSIVE (en-US)
intent-classification benchmark. This is the reference baseline the project
anchors against: running it end-to-end proves the core toolchain works
(torch + CUDA, transformers, datasets) and gives a real accuracy number.

Data is loaded via fedknob.data.massive (the single source-of-truth loader
that reads the official alexa/massive release), so this baseline and the
federated worker_id partitioning read identical data.

Usage:
    python train_baseline_massive.py --quick     # ~2k-example smoke run (1-2 min)
    python train_baseline_massive.py             # full run (~5 min on an 8 GB GPU)

    # security-oriented diagnostics:
    python train_baseline_massive.py --per-class-report   # per-class P/R/F1

What to expect:
    full    : ~85% test accuracy on MASSIVE en-US (60-way intent classification).

Metrics reported :
    accuracy            - headline number.
    balanced_accuracy   - average per-class recall; robust to MASSIVE's class imbalance.
    f1_macro            - per-class F1 averaged equally; catches rare-intent failures
                          that plain accuracy hides.
    f1_weighted         - average per-class F1 scores, each weighted by its number of samples
    precision_macro     - mean per-class precision,flags over-predicted (false positives) classes
    recall_macro        - mean per-class recall; under-predicted (false negatives / misses) classes
    top5_accuracy       - the true intent is among the model's top 5 ranked predictions
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import transformers
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# Loader for the Amazon MASSIVE dataset. The same module
# also provides the worker_id partitioning used in the federated phases, so the
# baseline and the federated clients read identical data. PROJECT_ROOT anchors
# all artifacts to the repo root regardless of the current working directory.
from fedknob.data.massive import PROJECT_ROOT, load_massive_en

MODEL_NAME = "distilbert-base-uncased"

transformers.logging.set_verbosity_error()

def build_model(num_labels: int):
    """Full DistilBERT sequence classifier -- all parameters trainable"""
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )


def compute_metrics(eval_pred):
    """Clean (benign) baseline metrics for 60-way multiclass intent classification.

    Macro-averaged metrics are important, because MASSIVE intents are
    imbalanced and accuracy alone will look healthy even if the model quietly drops
    a rare class.
    """
    logits, labels = eval_pred
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    preds = np.argmax(logits, axis=-1)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    # top-5 accuracy: was the true label among the model's 5 highest-scoring intents?
    k = min(5, logits.shape[-1])
    topk = np.argsort(logits, axis=-1)[:, -k:]
    top5_acc = float(np.mean([lab in row for lab, row in zip(labels, topk, strict=True)]))

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "top5_accuracy": top5_acc,
    }


def per_class_report(trainer, eval_ds, id2label, out_dir):
    """Per-class precision/recall/F1.

    Per-class RECALL is the early-warning signal a global accuracy number hides:
    a model can keep high overall accuracy while quietly dropping a rare intent.
    """
    pred = trainer.predict(eval_ds)
    logits = np.asarray(pred.predictions)
    labels = np.asarray(pred.label_ids)
    preds = np.argmax(logits, axis=-1)

    label_ids = sorted(id2label) if id2label else sorted(set(labels.tolist()))
    target_names = [str(id2label[i]) for i in label_ids] if id2label else None

    report = classification_report(
        labels, preds, labels=label_ids, target_names=target_names,
        zero_division=0, output_dict=True,
    )
    cm = confusion_matrix(labels, preds, labels=label_ids)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "per_class_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    np.savetxt(os.path.join(out_dir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

    rows = [
        (name, vals["recall"], int(vals["support"]))
        for name, vals in report.items()
        if name not in ("accuracy", "macro avg", "weighted avg")
    ]
    rows.sort(key=lambda r: r[1])
    print("\nLowest-recall intents (watch for imbalance issues):")
    for name, rec, sup in rows[:5]:
        print(f"  {name:<28} recall={rec:.3f}  support={sup}")
    print(f"\nSaved per-class report + confusion matrix to: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Full DistilBERT fine-tune baseline on MASSIVE (en-US)"
    )
    ap.add_argument("--quick", action="store_true", help="small subset smoke run")
    ap.add_argument("--epochs", type=float, default=10.0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5,
                    help="full fine-tuning LR (lower than the LoRA phases use)")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "artifacts" / "baseline"),
                    help="output dir (default: <repo root>/artifacts/baseline)")
    ap.add_argument("--per-class-report", action="store_true",
                    help="save per-class P/R/F1 + confusion matrix after training")
    args = ap.parse_args()

    print("Loading Amazon MASSIVE (en-US) ...")
    ds = load_massive_en("en-US")

    # MASSIVE stores `label` as the intent NAME (e.g. "weather_query").
    # Build a stable name->id map; fall back if a future version ships int labels.
    sample = ds["train"][0]["label"]
    if isinstance(sample, str):
        label_names = sorted({lab for split in ds for lab in ds[split]["label"]})
        label2id = {name: i for i, name in enumerate(label_names)}
        num_labels = len(label_names)
    else:
        label2id = None
        feat = ds["train"].features["label"]
        num_labels = getattr(feat, "num_classes", None) or int(max(ds["train"]["label"])) + 1
    id2label = {i: name for name, i in label2id.items()} if label2id else None
    print(f"  intents (labels): {num_labels}")
    print(f"  train/val/test  : {len(ds['train'])}/{len(ds['validation'])}/{len(ds['test'])}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(batch):
        enc = tok(batch["text"], truncation=True, max_length=64)
        if label2id is None:
            enc["labels"] = batch["label"]
        else:
            enc["labels"] = [label2id[lab] for lab in batch["label"]]
        return enc

    cols = ds["train"].column_names
    ds = ds.map(preprocess, batched=True, remove_columns=cols)

    train_ds, eval_ds = ds["train"], ds["test"]
    if args.quick:
        train_ds = train_ds.select(range(min(2000, len(train_ds))))
        eval_ds = eval_ds.select(range(min(1000, len(eval_ds))))
        print("  [quick mode] using a small subset for a fast smoke test")

    model = build_model(num_labels)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_train:,} / {n_total:,} (100% -- full fine-tune)")

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
        report_to=[],  # change to ["wandb"] after you run `wandb login`
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

    print("\n=== Clean baseline metrics (test) ===")
    for key in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
                "precision_macro", "recall_macro", "top5_accuracy"):
        ek = f"eval_{key}"
        if ek in metrics:
            print(f"  {key:<18}: {metrics[ek]:.4f}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "baseline_metrics.json"), "w") as f:
        json.dump({k: v for k, v in metrics.items() if k.startswith("eval_")}, f, indent=2)

    # --- optional diagnostics ---------------------------------------------------
    if args.per_class_report:
        per_class_report(trainer, eval_ds, id2label, args.out)

    model.save_pretrained(args.out)
    print(f"\nSaved fine-tuned model to: {args.out}")


if __name__ == "__main__":
    main()
