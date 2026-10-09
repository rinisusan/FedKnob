"""The shared metric suite -- one definition, used by every phase.

This lived in ``scripts/train_baseline_massive.py`` and was imported from there by
``src/fedknob/fl/task.py`` as a bare ``from train_baseline_massive import
compute_metrics``. That resolves only when ``scripts/`` happens to be on
``sys.path``, which the runner scripts arrange and nothing else does -- so the
federated evaluator could not be imported from a test, a notebook, or any new
entry point. Library code depending on a caller's path manipulation is the bug;
moving the function into the package is the fix.

``train_baseline_massive`` re-exports it, so the Week-1/2 scripts are unchanged
and the centralized baselines and every federated round still score through the identical
code path. That equality is the reason 88.70%, 88.06% and the per-round federated
numbers are comparable at all.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)


def compute_metrics(eval_pred):
    """Clean (benign) baseline metrics for 60-way multiclass intent classification.

    Macro-averaged metrics are important, because MASSIVE intents are
    imbalanced and accuracy alone will look healthy even if the model quietly
    drops a rare class.
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


