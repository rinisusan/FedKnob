# Why the Non-IID Knob Barely Turns at Real Client Boundaries

Research code for the ACM/IEEE SEC 2026 poster.

Federated learning benchmarks manufacture client heterogeneity with a Dirichlet
label-skew parameter α applied to individual samples. Edge deployments do not
have that freedom: a client is a device with an owner, and its data arrives
whole. This repository runs both partitioning schemes on the same corpus, over
the same α grid, scored by the same code, varying only the atomic unit.

**Three results.**

1. **α loses an order of magnitude of leverage** once clients are whole owners
   rather than arbitrary shards.
2. **Reported divergence is uncalibrated** — most of it, at fine partitions, is
   finite-sample artifact rather than skew.
3. **Structure, not magnitude, costs accuracy** at matched heterogeneity.

## Method

**Corpus.** Amazon MASSIVE 1.0, en-US training split: 11,514 utterances, 682
annotators, 60 intents. A `worker_id` identifies the author of each utterance,
giving real ownership boundaries rather than synthetic ones.

**The two schemes.** *Sample-level* applies the Dirichlet to individual
utterances, as in Hsu et al.; *owner-atomic* applies the same Dirichlet to whole
speakers, so a speaker's utterances are never split. Rows, per-client sizes and
the calibration floor are held identical — only the atomic unit changes.

**Metrics.** Effective intents per client, exp(H(P_c)), and mean
Jensen–Shannon divergence to the pooled label distribution — both scored
against a size-matched null in which every client is resampled i.i.d. from the
pooled distribution at its own size.

**Models and federated setup.** DistilBERT with a 60-way head; a rank-8 LoRA
adapter (147,456 persistent parameters) reaches 88.06% against 88.70% for the
full fine-tune. Federated runs use Flower 1.32.1 on Ray: FedAvg weighted by
client example counts, 10% participation, 2 local epochs, 30 rounds, evaluated
centrally on the untouched test split. The server warm-starts from a proxy
adapter fitted on the validation split — data disjoint from every client
partition, so it never sees client rows.

## Setup

```bash
# Python 3.11+. On NVIDIA 50-series / Blackwell GPUs install torch nightly first:
#   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
pip install -e ".[dev]"
python scripts/verify_env.py
```

## Reproducing the results

Every number in the submission comes from the commands below, in this order.
Steps 1–4 are **training-free and run in seconds**; only step 5 needs a GPU.

### 1. Build the three partition arms

```bash
# natural383 -- one speaker = one client, >=10 utterances. No alpha, no seed.
python scripts/partition_by_speaker_identity.py --min-utts 10 --calibrate

# shardm383 -- Dirichlet alpha=1.0, rows and per-client sizes copied from natural383
python scripts/partition_shard_matched.py --alpha 1.0 --tag shard383_matched --calibrate

# shard383 -- alpha=0.5, the field default
python scripts/partition_shard_matched.py --alpha 0.5 --tag shard383 --calibrate
```

### 2. Measure the partitions

```bash
python scripts/analyze_owner_concentration.py    # 6.52 vs 12.38, ratio 0.527
python scripts/sample_level_alpha_sweep.py       # 8.8x / 12.9x leverage lost
python scripts/calibrate_divergence.py           # the 69% finite-sample share
python scripts/sweep_partition_seeds.py          # the 0.44 partition-seed floor
python scripts/floor_vs_classes_per_sample.py    # K/n generality (CIFAR-10, FEMNIST)
```

### 3. Measure client structure

```bash
python scripts/analyze_client_structure.py       # common-mode fraction, 0.0011 vs 0.0291
```

### 4. Verify everything is self-consistent

```bash
python scripts/verify_repo.py                    # re-derives every reported number
pytest -q
```

`verify_repo.py` fails if any value asserted in prose is absent from an
artifact. Run it before trusting a figure.

### 5. The federated runs (GPU)

Five seeds per arm. All three use identical settings — only `--partition`
changes.

```bash
for SEED in 42 43 44 45 46; do
  python scripts/run_fedavg.py --partition artifacts/partitions/households_identity_min10.parquet \
      --clients 383 --rounds 30 --init proxy --mode ephemeral --rare-k 20 --seed $SEED \
      --out artifacts/fl/fedavg_proxy_natural383_r30_s$SEED.json

  python scripts/run_fedavg.py --partition artifacts/partitions/households_shard383_matched.parquet \
      --clients 383 --rounds 30 --init proxy --mode ephemeral --rare-k 20 --seed $SEED \
      --out artifacts/fl/fedavg_proxy_shardm383_r30_s$SEED.json

  python scripts/run_fedavg.py --partition artifacts/partitions/households_shard383.parquet \
      --clients 383 --rounds 30 --init proxy --mode ephemeral --rare-k 20 --seed $SEED \
      --out artifacts/fl/fedavg_proxy_shard383_r30_s$SEED.json
done
```

Defaults supply the rest of the configuration: `--fraction-fit 0.1`,
`--epochs 2`, `--lr 2e-4`, `--batch-size 32`, `pre_classifier` frozen,
`--eval-fraction 0.0` (evaluation is central, on the untouched test split).

## Repository layout

```
scripts/       partitioning, measurement, federated runs, figures, verification
src/fedknob/
  data/        MASSIVE loader; speaker-grouped Dirichlet partitioning
  models/      DistilBERT classifier and LoRA builders
  eval/        metric suite shared by the centralized and federated paths
  fl/          Flower client/server, FedAvg strategy, parameter handling
  utils/       seeding, console helpers
artifacts/
  baseline/    centralized metrics, model cards, LoRA adapters
  partitions/  frozen partitions (.parquet) + measurement reports (.json)
  fl/          one JSON per federated run: config, provenance, per-round history
  figures/     all generated figures (.png and .pdf)
tests/         unit and integration tests
```

`artifacts/README.md` documents what each artifact contains and which script
writes it.

## Notes

- **Dataset source.** Read directly from the official `alexa/massive` 1.0
  release (JSONL tarball) by `src/fedknob/data/massive.py`, so every
  experiment reads identical data.
- **Determinism.** Partitions are frozen to disk and byte-identical on re-run
  (`--check-byte-identical`). Speaker-identity partitions involve no randomness
  and therefore no seed.
- **Test split.** Never enters a client partition; it serves only as the
  central evaluation set.
- **GPU.** Validated on an NVIDIA RTX 5060 (Blackwell, sm_120) with the torch
  nightly cu128 wheel.
