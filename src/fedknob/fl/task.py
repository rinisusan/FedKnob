"""Local training and evaluation -- one client's turn, and one model's score.

Deliberately a plain PyTorch loop rather than HuggingFace ``Trainer``. A Week-4
run is 300+ client turns; constructing a ``Trainer`` per turn costs seconds of
setup, writes checkpoint and logging directories nobody reads, and hides the step
count. Here the loop is ~20 lines and the step count is returned, which the exit
criteria need.

Metrics come from ``fedknob.eval.metrics.compute_metrics`` -- the same
function that produced 88.70% and 88.06% -- so federated numbers are directly
comparable to the centralized ones.

**One deliberate difference from the baselines: evaluation runs in fp32.** The
Week-1/2 runs evaluated under fp16 autocast. Accuracy is a count, so a borderline
utterance can flip between precisions: one example is 0.034 percentage points at
n=2,974, which is visible in the fourth decimal. fp32 is reproducible across
machines, and the round-0 gate therefore compares to 0.8806 within a couple of
examples rather than demanding bit-equality.
"""

from __future__ import annotations

BATCH_SIZE = 32  # centralized recipe
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 2e-4  # centralized recipe: 10x the full-fine-tune lr


def _device(explicit=None):
    import torch

    if explicit is not None:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loader(ds, batch_size: int, shuffle: bool, seed: int | None = None):
    """DataLoader over a tokenized HF Dataset.

    ``drop_last`` is explicitly False. At N=200 the smallest household holds 30
    utterances against a batch size of 32 -- dropping the partial batch would
    give that client zero gradient steps while still reporting 30 examples to
    the aggregator, so it would contribute weight without contributing learning.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, DataCollatorWithPadding

    from fedknob.fl.data import TOKENIZER_NAME

    collate = DataCollatorWithPadding(AutoTokenizer.from_pretrained(TOKENIZER_NAME))
    gen = None
    if shuffle and seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
    return DataLoader(
        ds.with_format("torch"),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        drop_last=False,
        generator=gen,
    )


def train_one_client(
    model,
    dataset,
    epochs: int = 2,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    seed: int = 42,
    device=None,
) -> dict:
    """Train ``model`` in place on one client's data. Returns a turn report.

    ``first_batch_loss`` and ``last_batch_loss`` exist for exit criterion E7:
    a client whose loss does not fall did not learn, and that failure is
    otherwise indistinguishable from "learned but averaging destroyed it".
    """
    import torch

    dev = _device(device)
    model.to(dev).train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    loader = _loader(dataset, batch_size, shuffle=True, seed=seed)

    losses: list[float] = []
    for _ in range(epochs):
        for batch in loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(out.loss.detach()))

    if not losses:
        raise RuntimeError(
            f"client produced 0 gradient steps from {len(dataset)} examples "
            f"(batch_size={batch_size}). Check drop_last."
        )
    return {
        "n_examples": len(dataset),
        "n_steps": len(losses),
        "mean_loss": sum(losses) / len(losses),
        "first_batch_loss": losses[0],
        "last_batch_loss": losses[-1],
        "loss_fell": losses[-1] < losses[0],
    }


#: The intent whose recall is tracked separately -- ``iot_wemo_off``.
DEFAULT_TARGET_INTENT = 28

#: How many of the rarest intents form the group comparator.
#:
#: The target alone is ~18 of the 2,974 test utterances -- one example is 5.6
#: points, and across three seeds its measured decay spread 11 points. Unusable
#: on its own.
#:
#: k=10 pools 81 examples (1 example = 1.23 points) and still spread 8.6 points
#: across seeds. k=20 pools roughly 250, cutting proportional noise by ~1.8x,
#: which is what makes a claim about how fast the federation forgets rare
#: intents decidable at all. The 20 rarest of 60 intents
#: still average ~12 test examples each, so the group is genuinely the tail.
DEFAULT_RARE_K = 20


def rare_intent_ids(labels, k: int, exclude: int | None = None) -> list[int]:
    """The k least frequent intents in ``labels``, ascending by count.

    Derived from the evaluation set rather than hard-coded, so it stays correct
    if the split or the label map changes.

    ``exclude`` drops the separately-tracked intent *before* taking the k
    rarest. This is not cosmetic: if the tracked intent sat inside the
    comparator, a change in that one intent would move the comparator with it.
    The comparator has to measure knowledge the tracked intent does not touch.

    Until k=20 this held only by accident: at k=10 the target (18 examples) was
    commoner than the 10 rarest, so it fell outside. At k=20 it would land
    inside. Excluding it explicitly makes k a free parameter and removes a
    coincidence the result was resting on.
    """
    import numpy as np

    ids, counts = np.unique(np.asarray(labels), return_counts=True)
    if exclude is not None:
        keep = ids != exclude
        ids, counts = ids[keep], counts[keep]
    return [int(i) for i in ids[np.argsort(counts, kind="stable")][:k]]


def _recall(labels, preds, wanted) -> tuple[float, int]:
    """Recall over the union of ``wanted`` classes, and its support.

    Pooled, not averaged per class: with ~18 examples in some classes a per-class
    mean is dominated by whichever tiny class happened to flip. Support is
    returned alongside because a recall figure without its n cannot be read.
    """
    import numpy as np

    mask = np.isin(np.asarray(labels), list(wanted))
    n = int(mask.sum())
    if n == 0:
        return float("nan"), 0
    return float((np.asarray(preds)[mask] == np.asarray(labels)[mask]).mean()), n


def evaluate(
    model,
    dataset,
    batch_size: int = EVAL_BATCH_SIZE,
    device=None,
    target_intent: int = DEFAULT_TARGET_INTENT,
    rare_k: int = DEFAULT_RARE_K,
) -> dict:
    """Score ``model`` on ``dataset``. fp32, no autocast -- see the module note.

    Beyond the shared metric suite this returns three things the federated phases
    need and the centralised baselines did not: the evaluation ``loss`` (the
    server logged a hard-coded 0.0 without it), recall on the tracked intent, and
    recall pooled over the rarest intents -- the tail the global accuracy number
    hides, and what the owner-versus-shard comparison is read on.
    """
    import numpy as np
    import torch

    from fedknob.eval.metrics import compute_metrics

    dev = _device(device)
    model.to(dev).eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in _loader(dataset, batch_size, shuffle=False):
            labels = batch.pop("labels")
            batch = {k: v.to(dev) for k, v in batch.items()}
            all_logits.append(model(**batch).logits.float().cpu().numpy())
            all_labels.append(labels.numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    preds = logits.argmax(-1)

    metrics = compute_metrics((logits, labels))
    metrics["n_examples"] = int(len(labels))
    metrics["n_correct"] = int((preds == labels).sum())
    # The raw predictions travel with the metrics so a caller can compute a
    # further statistic out of THIS pass instead of running a second one over
    # the same 2,974 rows. Not a scalar, so the artifact writer -- which copies
    # named fields -- never sees it.
    metrics["preds"] = preds

    # Cross-entropy in float64: the sum runs over ~3k terms and the value is
    # compared across rounds at four decimals.
    z = logits.astype(np.float64)
    z -= z.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    metrics["loss"] = float(-logp[np.arange(len(labels)), labels].mean())

    r, n = _recall(labels, preds, [target_intent])
    metrics["target_recall"], metrics["target_support"] = r, n
    metrics["target_intent"] = int(target_intent)

    rare = rare_intent_ids(labels, rare_k, exclude=target_intent)
    r, n = _recall(labels, preds, rare)
    metrics["rare_recall"], metrics["rare_support"] = r, n
    metrics["rare_intents"] = rare
    return metrics


def predict(model, dataset, batch_size: int = EVAL_BATCH_SIZE, device=None):
    """Argmax predictions only -- no metrics, no labels needed.

    Returns nothing but "which intent did the model say" for each row. True
    labels are carried in the dataset but are not the question here
    to the target, so accuracy against the true label is not the statistic.
    """
    import numpy as np
    import torch

    dev = _device(device)
    model.to(dev).eval()
    out = []
    with torch.no_grad():
        for batch in _loader(dataset, batch_size, shuffle=False):
            batch.pop("labels", None)
            batch = {k: v.to(dev) for k, v in batch.items()}
            out.append(model(**batch).logits.float().argmax(-1).cpu().numpy())
    return np.concatenate(out)


def accuracy(model, dataset, batch_size: int = EVAL_BATCH_SIZE, device=None) -> float:
    """Just the headline number, for per-round logging."""
    m = evaluate(model, dataset, batch_size, device)
    return float(m.get("accuracy", m.get("eval_accuracy")))
