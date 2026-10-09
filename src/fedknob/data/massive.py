"""Amazon MASSIVE dataset loader + federated partitioning helpers.

Read the Amazon MASSIVE dataset directly from the official alexa/massive release JSONL.
Two views of the SAME data:
    load_locale(...)      -> raw pandas DataFrame (incl. worker_id) for partitioning
    load_massive_en(...)  -> HuggingFace DatasetDict (text/label) for training

Source release: https://github.com/alexa/massive
Tarball layout: 1.0/data/{locale}.jsonl
Each line has:  id, locale, partition, scenario, intent, utt, annot_utt,
                worker_id, slot_method, judgments.

Usage:
    # training (baseline / LoRA)
    ds = load_massive_en("en-US")            # DatasetDict: train/validation/test

    # inspecting the raw records, including worker_id
    df = load_locale("en-US", partition="train")

The federated phases do NOT partition here. They read the frozen partitions built
by the partitioning scripts, which are byte-identical from seed 42 and are the reference every
result is tied to:

    from fedknob.data.partition import load_partition
    clients = load_partition("artifacts/partitions/households_100.parquet")

Building a partition on the fly instead would produce different clients from the
ones every committed measurement describes, so this module deliberately offers no
partitioning helper of its own.
"""

from __future__ import annotations

import io
import json
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

MASSIVE_URL = (
    "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz"
)


def _find_project_root(start: Path) -> Path:
    """From `start` to the FedKnob repo root (dir with pyproject.toml
    or .git)."""
    p = start.resolve()
    for parent in (p, *p.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return Path.cwd()


# Repo root, resolved from this file's location (src/fedknob/data/massive.py).
PROJECT_ROOT = _find_project_root(Path(__file__))
CACHE_DIR = PROJECT_ROOT / "artifacts" / "massive_raw"

# MASSIVE 'partition' value -> HuggingFace split name
_SPLIT_MAP = {"train": "train", "dev": "validation", "test": "test"}


def download_massive(url: str = MASSIVE_URL, cache_dir: Path = CACHE_DIR) -> Path:
    """Download the MASSIVE tarball once and cache it locally."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    tar_path = cache_dir / "amazon-massive-dataset-1.0.tar.gz"
    if not tar_path.exists():
        print(f"Downloading Amazon MASSIVE from {url} ...")
        urllib.request.urlretrieve(url, tar_path)
    return tar_path


def load_locale(
    locale: str = "en-US",
    partition: str | None = None,
    cache_dir: Path = CACHE_DIR,
) -> pd.DataFrame:
    """Return one locale's records as a raw DataFrame (incl. worker_id).

    partition: keep only 'train' / 'dev' / 'test' rows, or None for all.
    Columns are the original MASSIVE fields (utt, intent, worker_id, ...).
    """
    tar_path = download_massive(cache_dir=cache_dir)
    member = f"1.0/data/{locale}.jsonl"
    rows = []
    with tarfile.open(tar_path, "r:gz") as tar:
        f = tar.extractfile(member)
        if f is None:
            raise FileNotFoundError(f"{member} not found in tarball")
        for line in io.TextIOWrapper(f, encoding="utf-8"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if partition is not None:
        df = df[df["partition"] == partition].reset_index(drop=True)
    return df


def load_massive_en(locale: str = "en-US", cache_dir: Path = CACHE_DIR):
    """Return a HuggingFace DatasetDict for training (used by the trainer).

    Built from the same official JSONL as load_locale, so the baseline, the LoRA
    variant, and the federated clients all read identical data. Splits:
    train/validation/test (MASSIVE's train/dev/test). Columns:
        text       -> the utterance   (from 'utt')
        label      -> the intent name (from 'intent', e.g. "weather_query")
        worker_id  -> annotator id    (carried along; unused by single-machine training)
    """
    from datasets import Dataset, DatasetDict

    df = load_locale(locale, partition=None, cache_dir=cache_dir)
    splits = {}
    for raw, name in _SPLIT_MAP.items():
        sub = df[df["partition"] == raw]
        splits[name] = Dataset.from_dict(
            {
                "text": sub["utt"].tolist(),
                "label": sub["intent"].tolist(),
                "worker_id": sub["worker_id"].tolist(),
            }
        )
    return DatasetDict(splits)


def build_label_map(intents) -> tuple[dict[str, int], int]:
    """Stable intent-name -> id map (sorted), shared by training and partitioning.

    `intents` may be any iterable of intent-name strings (e.g. df['intent'] or a
    DatasetDict split's 'label' column). Returns (label2id, num_labels).
    """
    names = sorted(set(intents))
    return {name: i for i, name in enumerate(names)}, len(names)


def describe_workers(df: pd.DataFrame) -> None:
    """Print how many distinct workers exist and the utterances-per-worker spread."""
    counts = df.groupby("worker_id").size()
    print(f"rows: {len(df)}")
    print(f"distinct worker_id: {counts.size}")
    print(f"utts/worker -> min {counts.min()}, median {int(counts.median())}, max {counts.max()}")


if __name__ == "__main__":
    # quick demo / sanity check
    df = load_locale("en-US", partition="train")
    print(df[["id", "intent", "worker_id", "utt"]].head())
    describe_workers(df)
    print("\nFederated clients come from the frozen Week-3 partitions, not from here:")
    print("  from fedknob.data.partition import load_partition")
    print("  load_partition('artifacts/partitions/households_100.parquet')")
