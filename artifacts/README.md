# Artifacts

Reference artifacts. Large transient artifacts (`state/`, `runs/`, `sweeps/`,
`tokenized/`) are gitignored, as are model weights; metrics and model cards are
committed.

## `baseline/`

Centralized full fine-tune and rank-8 LoRA: metrics, per-class reports,
confusion matrices, model cards. Adapter weights are gitignored.

## `partitions/`

**Partition files.** Long-form parquet, one row per utterance, columns
`(household_id, row_index, worker_id, intent, utt)`. Every measurement script
reads this schema, so the two partitioning schemes are interchangeable as inputs.

| file | scheme | clients |
|---|---|---|
| `households_{20,50,100,200}.parquet` | speaker-atomic Dirichlet(α=0.5), size-banded, seed 42 | 20 / 50 / 100 / 200 |
| `households_identity_all.parquet` | one speaker = one client, unfiltered, **no α, no seed** | 682 |
| `households_identity_min10.parquet` | the same, ≥10 utterances (eligibility rule) | 383 |

`*_stats.json` accompanies each with its verification report.

`households_identity_all` is the baseline with no decisions in it — every speaker,
every utterance. It is measurement-only: at a median of 12 utterances per client,
an 80/20 split leaves about two test samples, so per-client accuracy would be
quantised to {0, 0.5, 1}. `min10` is the trainable variant, and 187 clients under
five utterances is what motivates the threshold.

**Measurements.** All training-free.

| file | produced by | what it holds |
|---|---|---|
| `divergence_report.json` | `analyze_partition_divergence.py` | raw mean JSD to global, pairwise JSD, effective intents |
| `calibrated_divergence.json` | `calibrate_divergence.py` | size-matched floor and observed/floor ratio per partition |
| `households_identity_min10_calibrated.json` | `partition_by_speaker_identity.py --calibrate` | the same, for the identity partition |
| `owner_concentration.json` | `analyze_owner_concentration.py` | per-speaker label breadth against a matched-size reference |
| `granularity_contrast.json` | `sample_level_alpha_sweep.py` | α swept at sample vs owner granularity, 5 α × 3 seeds |
| `seed_sweep.json` | `sweep_partition_seeds.py` | partition-seed stability, seeds 42–46 × 4 client counts |

### Reading the divergence numbers

Raw JSD is **not** interpretable on its own. It is biased upward at small sample
sizes: a client holding 26 utterances cannot cover 60 intents, so it looks
divergent from the pooled distribution however it was produced. The bias grows as
clients shrink, in the same direction as the heterogeneity one wants to claim.

`calibrated_divergence.json` therefore reports, per partition, what the same
measurement returns for a reference partition whose clients are **drawn at random
from the pooled distribution at matched sizes** — a partition with no systematic
heterogeneity in it at all. Call that the *floor*.

> Three separate things are described as "zero" in this project and only the
> first is:
>
> - the reference partition's **systematic** heterogeneity is zero by
>   construction;
> - its **measured** divergence is not — 0.0243 at 582 utterances per client,
>   0.3865 at 26, 0.5553 at 12;
> - the identity partition's **seed variance** is exactly zero, which is
>   determinism, a different property entirely.
>
> "Indistinguishable from chance" shows up as a ratio near **1.0**, never as a
> divergence near 0.

| partition | clients | median utts | raw JSD | floor | **ratio** |
|---|---|---|---|---|---|
| Dirichlet N=20 | 20 | 582 | 0.0509 | 0.0243 | **2.10×** |
| Dirichlet N=50 | 50 | 218 | 0.1185 | 0.0659 | **1.80×** |
| Dirichlet N=100 | 100 | 108 | 0.2117 | 0.1350 | **1.57×** |
| Dirichlet N=200 | 200 | 55 | 0.3255 | 0.2243 | **1.45×** |
| identity ≥10 | 383 | 26 | 0.5010 | 0.3865 | **1.30×** |
| identity, all | 682 | 12 | 0.6478 | 0.5553 | **1.17×** |

`calibrated_divergence.json` also stores `null_share_of_raw` — the floor as a
fraction of the raw value, 48% at N=20 rising to 86% at one client per speaker.
It is exactly 1/ratio, so it is not tabulated here.

The two right-hand columns move in opposite directions across every row: raw JSD
rises 12.7× down the table while the calibrated ratio falls from 2.10× to 1.17×.
Read the raw column alone and heterogeneity appears to climb steeply with client
count; read the calibrated one and it declines. Same partitions, same code.

Report the **ratio**. The floor is not transferable between rows — it is a
function of client size, and reusing one value across partitions of different
granularity inflates the fine-grained ones by roughly an order of magnitude.

All values are means over clients. Observed and floor must use the same statistic
or the correction is meaningless; medians run a few percent lower and change
nothing.

The identity row is a different scheme, not a fifth point on a curve: it has no
Dirichlet step and no size band. Adjusted for client size it sits slightly above
the constructed partitions rather than below, so real client boundaries produce
marginally more genuine skew — while both schemes remain dominated by the size
effect.

## `figures/`

All generated from the JSON above; nothing is hand-edited, so regenerating cannot
disagree with the data. Greyscale-safe — fills, line styles and marker shapes
carry every distinction, no figure depends on colour.

| file | generator | reads | shows |
|---|---|---|---|
| `fig_alpha_main` / `fig_calibration_main` | `make_leverage_figures.py` | `granularity_contrast.json`, `calibrated_divergence.json` | the two headline figures; **Dirichlet partitions only** |
| `fig_rare_trajectory.{png,pdf}` | `make_partition_figures.py` | `fl/fedavg_proxy_{arm}_r30_s*.json` | rare-intent recall over 30 rounds, three arms, mean of 5 seeds |

Regenerate everything with one command:

```bash
python scripts/make_all_figures.py            # every figure, from the JSON
python scripts/make_all_figures.py --check    # verify only, write nothing
```

Each generator is still runnable on its own; `make_all_figures.py` orchestrates
and contains no plotting logic. It reports missing inputs as a data problem
rather than a traceback, and `--check` is the pre-commit gate — three separate
scripts write here, and running two of them leaves the third silently disagreeing
with the table beside it.

Label heatmaps, pairwise-JSD maps and the seed-sweep plot are side effects of
the analysis scripts that compute them. They are not shipped; the scripts create
`reports/` on demand:

```bash
python scripts/partition_households.py
python scripts/analyze_partition_divergence.py --heatmap
python scripts/sweep_partition_seeds.py
```

`fig_calibration_main` is the one to read first — it shows raw divergence and
its size-matched floor moving in opposite directions as clients narrow. It
deliberately covers the four constructed Dirichlet partitions only, since that
is what the submission compares; the identity partitions are omitted.

## `massive_raw/`

The official `alexa/massive` 1.0 JSONL tarball, read directly by
`src/fedknob/data/massive.py`. Committed before the ignore rule was added, so
it remains tracked.

## Federated runs

`fl/` holds one JSON per FedAvg run. Each records its full config, the
init provenance (which checkpoint the server warm-started from, and proof it
never saw client rows), and a per-round history including accuracy, macro-F1
and rare-intent recall.

The three arms behind the SEC 2026 result, five seeds each:

| file pattern | partition | what it is |
|---|---|---|
| `fedavg_proxy_natural383_r30_s4[2-6].json` | `households_identity_min10.parquet` | one speaker = one client |
| `fedavg_proxy_shardm383_r30_s4[2-6].json` | `households_shard383_matched.parquet` | Dirichlet α=1.0, size- and excess-matched |
| `fedavg_proxy_shard383_r30_s4[2-6].json` | `households_shard383.parquet` | Dirichlet α=0.5, field default |

All three hold identical rows (10,316), identical per-client sizes and an
identical 0.3865 floor. Only the structure differs.

`make_partition_figures.py` globs `fedavg_proxy_{arm}_r30_s*.json`, so removing
a seed silently changes the n of the reported means rather than failing.

### Which runs back the α-versus-seed comparison

The reported pair -- α moves macro-F1 by **1.12** points against **0.95** for a
seed change -- comes from the N=100 runs written *before* the `server_round`
fix in `fl/server.py`:

| quantity | files |
|---|---|
| α span, 1.12 | `fedavg_proxy_a005_n100_f01_r30.json`, `fedavg_proxy_a1_n100_f01_r30.json` |
| seed span, 0.95 | `fedavg_proxy_ephemeral_n100_f01_r30.json`, `..._s43.json`, `..._s44.json` |

`fedavg_proxy_ephemeral_n100_f01_r30_roundfix_s4[2-6].json` are a later,
five-seed set run *after* that fix, and they give a seed span of **0.67**.

Both numbers are correct; they are not interchangeable. Before the fix every
client replayed the same batch order in every round, so pre-fix and post-fix
runs are not comparable. No post-fix α sweep exists, so quoting 0.67 against
1.12 would compare across the fix. The pair is held on the same side of it.
