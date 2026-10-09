"""Frozen partition parquet -> per-client tokenized datasets.

Two rules are enforced here rather than left to convention, because breaking
either produces plausible-looking accuracy instead of an error.

**1. The parquet is self-contained.** Its columns are ``household_id, row_index,
worker_id, intent, utt`` -- the text and the intent name travel with the
assignment, so tokenise ``utt`` directly. Do NOT use ``row_index`` to index back
into ``load_massive_en()``: that column indexes ``load_locale(partition="train")``,
and if the two ever order differently every utterance is silently mislabelled.

**2. One label map, shared with the baselines.** ``build_label_map`` is applied to
``load_massive_en()["train"]["label"]`` -- the same call the centralized
baselines made. That ``classifier`` was trained against those exact 60 indices and the frozen
``pre_classifier`` feeds it. Deriving a fresh map from a partition would order the
intents differently (and a partition need not contain all 60), pointing the
checkpoint at the wrong classes.

Evaluation for vanilla FedAvg is **central**: the global model is scored on the
untouched MASSIVE test split, so no per-client split is needed and every client
trains on all its data. ``eval_fraction`` exists for the personalised phases,
where per-client accuracy is the statistic of interest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MAX_LENGTH = 64  # identical to the centralized baselines
TOKENIZER_NAME = "distilbert-base-uncased"
REQUIRED_COLUMNS = ("household_id", "worker_id", "intent", "utt")


@dataclass(frozen=True)
class ClientData:
    """One federated client. ``eval`` is None under central evaluation."""

    client_id: int
    train: object
    eval: object | None
    n_train: int
    n_eval: int
    n_speakers: int


def split_indices(n: int, eval_fraction: float, seed: int, client_id: int):
    """Deterministic per-client train/eval split. Pure -- no torch, no tokenizer.

    Seeded with ``seed + client_id`` so a client's held-out rows are stable across
    runs *and* independent of which other clients exist or of the round schedule.
    A client that appears in two different partitions still splits reproducibly.

    ``eval_fraction <= 0`` means central evaluation: everything trains.
    """
    import numpy as np

    if n <= 0:
        raise ValueError(f"client {client_id} has no rows")
    if eval_fraction <= 0.0:
        return np.arange(n), np.empty(0, dtype=int)
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError(f"eval_fraction must be in (0, 1), got {eval_fraction}")

    k = max(1, int(round(n * eval_fraction)))
    if k >= n:
        raise ValueError(
            f"client {client_id}: eval_fraction={eval_fraction} would take {k} of "
            f"{n} rows, leaving no training data."
        )
    perm = np.random.default_rng(seed + client_id).permutation(n)
    return perm[k:], perm[:k]


def load_label_map() -> tuple[dict[str, int], int]:
    """The shared label map -- single source of truth for all 60 intent indices."""
    from fedknob.data.massive import build_label_map, load_massive_en

    return build_label_map(load_massive_en()["train"]["label"])


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def _encode(texts, labels, tok, label2id):
    """Tokenise into a HuggingFace Dataset with the columns Trainer expects."""
    from datasets import Dataset

    unknown = sorted({x for x in labels if x not in label2id})
    if unknown:
        raise KeyError(
            f"{len(unknown)} intent(s) absent from the shared label map: "
            f"{unknown[:5]}. The partition and the label map disagree, which "
            f"would mislabel every affected utterance."
        )
    enc = tok(list(texts), truncation=True, max_length=MAX_LENGTH)
    enc["labels"] = [label2id[x] for x in labels]
    return Dataset.from_dict(enc)


def load_clients(
    partition_path: str | Path,
    eval_fraction: float = 0.0,
    seed: int = 42,
) -> tuple[dict[int, ClientData], dict[str, int]]:
    """Return ``{household_id: ClientData}`` plus the shared label map."""
    from fedknob.data.massive import PROJECT_ROOT
    from fedknob.data.partition import load_partition

    path = Path(partition_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"partition not found: {path}")

    df = load_partition(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} is missing column(s) {missing}")

    label2id, _ = load_label_map()
    tok = _tokenizer()

    # Pass 1 fixes every client's train/eval split, and therefore n_train.
    staged: dict[int, tuple] = {}
    for cid, sub in df.groupby("household_id", sort=True):
        texts, labels = sub["utt"].tolist(), sub["intent"].tolist()
        tr, ev = split_indices(len(texts), eval_fraction, seed, int(cid))
        staged[int(cid)] = (texts, labels, tr, ev, int(sub["worker_id"].nunique()))

    # Pass 2 encodes.
    clients: dict[int, ClientData] = {}
    for cid, (texts, labels, tr, ev, n_speakers) in staged.items():
        tr_texts = [texts[i] for i in tr]
        tr_labels = [labels[i] for i in tr]

        clients[int(cid)] = ClientData(
            client_id=int(cid),
            train=_encode(tr_texts, tr_labels, tok, label2id),
            eval=(
                _encode([texts[i] for i in ev], [labels[i] for i in ev], tok, label2id)
                if len(ev)
                else None
            ),
            n_train=len(tr),
            n_eval=len(ev),
            n_speakers=n_speakers,
        )

    verify_clients(
        clients,
        n_rows=len(df),
        n_households=df["household_id"].nunique(),
        eval_fraction=eval_fraction,
    )
    return clients, label2id


def load_central_eval(split: str = "test"):
    """The untouched MASSIVE split the global model is scored on.

    Only the *training* split is partitioned; dev and test are left whole, which
    is what makes this a clean yardstick against the 88.06% centralized number.
    """
    from fedknob.data.massive import load_massive_en

    name = {"test": "test", "dev": "validation", "validation": "validation"}[split]
    ds = load_massive_en()[name]
    label2id, _ = load_label_map()
    return _encode(ds["text"], ds["label"], _tokenizer(), label2id)


def verify_clients(clients, n_rows: int, n_households: int, eval_fraction: float) -> None:
    """Coverage and leakage checks. Pure -- callable from tests without torch."""
    total = sum(c.n_train + c.n_eval for c in clients.values())
    if total != n_rows:
        raise AssertionError(
            f"client rows sum to {total} but the partition has {n_rows}; "
            f"utterances were lost or duplicated in the split."
        )
    if len(clients) != n_households:
        raise AssertionError(f"{len(clients)} clients built from {n_households} household_ids")
    if eval_fraction <= 0.0 and any(c.n_eval for c in clients.values()):
        raise AssertionError("eval_fraction=0 but some client has eval rows")
    empty = sorted(c.client_id for c in clients.values() if c.n_train == 0)
    if empty:
        raise AssertionError(f"client(s) {empty} have no training data")


def summarise(clients) -> dict:
    """Shape of the federation, for the run manifest."""
    import numpy as np

    tr = np.array([c.n_train for c in clients.values()])
    sp = np.array([c.n_speakers for c in clients.values()])
    ev = np.array([c.n_eval for c in clients.values()])
    return {
        "n_clients": len(clients),
        "total_train": int(tr.sum()),
        "train_per_client": {
            "min": int(tr.min()),
            "median": int(np.median(tr)),
            "max": int(tr.max()),
        },
        "speakers_per_client": {
            "min": int(sp.min()),
            "median": int(np.median(sp)),
            "max": int(sp.max()),
        },
        "eval_per_client_median": int(np.median(ev)),
    }
