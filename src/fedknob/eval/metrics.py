"""The shared metric suite -- one definition, used by every phase.

This lived in ``scripts/train_baseline_massive.py`` and was imported from there by
``src/fedknob/fl/task.py`` as a bare ``from train_baseline_massive import
compute_metrics``. That resolves only when ``scripts/`` happens to be on
``sys.path``, which the runner scripts arrange and nothing else does -- so the
federated evaluator could not be imported from a test, a notebook, or any new
entry point. Library code depending on a caller's path manipulation is the bug;
moving the function into the package is the fix.

``train_baseline_massive`` re-exports it, so the Week-1/2 scripts are unchanged
and Week 1, Week 2 and every federated round still score through the identical
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
    imbalanced and accuracy alone will look healthy even if the model quietly drops
    a rare class -- exactly the failure mode a backdoored adapter can hide behind.
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


def mcnemar(clean_hit, trig_hit) -> dict:
    """Paired comparison of two binary outcomes measured on the *same* items.

    Both arguments are boolean "did the model predict the target intent"
    vectors -- one from untriggered text, one from the same utterances with the
    trigger inserted. Because they come from the same rows the two proportions
    are **paired**, and an independent-samples interval would be the wrong one
    (and conservative). McNemar's test uses only the discordant pairs:

        b = clean predicted the target, triggered did not
        c = clean did not,              triggered did
        delta = (c - b) / n

    ``delta`` is the statistic the whole attack rests on. Raw ASR is not:
    ``iot_wemo_off`` is 0.61% of the test split, so a calibrated clean model
    predicts it ~0.6% of the time on untriggered text, and the old model card's
    comparison against a uniform 1/60 made a null result look like a 3x anomaly.
    A trigger is admissible when ``delta`` is within noise of zero on a clean
    model -- that is what licenses attributing any later rise to the backdoor
    rather than to a pre-existing lexical bias.

    Lives here rather than in ``scripts/asr_baseline.py`` for the same reason
    ``compute_metrics`` does: the federated evaluator must not import from
    ``scripts/``, and the centralised control and the federated ASR have to be
    computed by identical code or the two numbers are not comparable.

    Variance for the difference of paired proportions follows Agresti (2002),
    eq. 10.4; the chi-square is continuity-corrected.
    """
    import math

    clean_hit = np.asarray(clean_hit, dtype=bool)
    trig_hit = np.asarray(trig_hit, dtype=bool)
    if clean_hit.shape != trig_hit.shape:
        raise ValueError(
            f"paired vectors must be the same length, got {clean_hit.shape} and "
            f"{trig_hit.shape} -- these are not the same utterances."
        )

    n = int(clean_hit.size)
    if n == 0:
        return {"n": 0, "b": 0, "c": 0, "delta": 0.0, "se": 0.0, "ci": [0.0, 0.0], "chi2": 0.0}

    b = int(np.sum(clean_hit & ~trig_hit))
    c = int(np.sum(~clean_hit & trig_hit))
    delta = (c - b) / n
    var = (b + c - (c - b) ** 2 / n) / (n**2)
    se = math.sqrt(max(var, 0.0))
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    return {
        "n": n,
        "b": b,
        "c": c,
        "delta": float(delta),
        "se": float(se),
        "ci": [float(delta - 1.96 * se), float(delta + 1.96 * se)],
        "chi2": float(chi2),
    }
