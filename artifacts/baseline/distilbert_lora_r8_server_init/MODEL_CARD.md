# Model Card -- FedKnob server-side proxy init (DistilBERT / MASSIVE)

## Summary

Rank-8 LoRA fine-tune of `distilbert-base-uncased` for 60-way intent
classification on Amazon MASSIVE (en), fitted on the **validation** split only.

This is the server-side initialisation for the federated phases. The federated
clients are built from the *training* split, so an adapter fitted on that split
has already seen every client's private data and cannot be a legitimate FL
starting point -- the server would have had to centralise the very data
federation exists to keep local. Fitting on a small held-out split instead makes
the round-0 model something a server could genuinely hold. Expect materially
lower standalone accuracy than the client-side baseline: that gap is the headroom the
federated rounds are supposed to close, and closing it is the result.

- **Fitted on:** MASSIVE en-US `validation` split -- server-side proxy init, disjoint from the client partitions
- **Evaluated on:** MASSIVE en-US `test` split (never fitted on by any phase)

- **Base model:** distilbert-base-uncased (66M params, frozen)
- **Task:** sequence classification (60 MASSIVE intents)
- **Method:** PEFT LoRA -- r=8, alpha=16, dropout=0.05,
  target_modules=['q_lin', 'v_lin'], modules_to_save=['pre_classifier', 'classifier']
- **Head regime:** `pre_classifier` + `classifier` both trainable
- **Dataset:** Amazon MASSIVE en-US via `fedknob.data.massive`
  (official alexa/massive 1.0 release)
- **Tokenization:** DistilBERT tokenizer, max_length=64 (cached to disk)
- **License:** Apache-2.0

## Parameter efficiency

| component | params | % of total |
|---|---|---|
| LoRA A/B matrices (persistent per-household adapter) | 147,456 | 0.22% |
| classification head (modules_to_save; shared, new init) | 636,732 | -- |
| total trainable | 784,188 | 1.16% |
| total | 67,783,800 | 100% |

The <1% feasibility claim refers to the LoRA matrices proper -- the bytes that
persist on-device per household. The classification head belongs to the shared
global model, not the per-household adapter.

### Head regimes

`build_lora_model()` supports two configurations of the classification head.

| regime | `pre_classifier` | `classifier` | trainable |
|---|---|---|---|
| centralised **(this card)** | trainable | trainable | 784,188 |
| federated | frozen, loaded from `artifacts/baseline/distilbert_lora_r8` | trainable, aggregated | 193,596 |

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

- Learning rate: 0.0002 (LoRA; 10x the full-fine-tune lr)
- Epochs: 25.0, batch size 32, max sequence length 64
- Optimizer/schedule: HF `Trainer` defaults (AdamW); fp16 when CUDA available
- Seed: 42

Reproduce with:

```bash
python scripts/train_lora_massive.py --train-split validation \
    --out artifacts/baseline/distilbert_lora_r8_server_init
```

## Evaluation (MASSIVE test split)

- accuracy: 0.8292
- full metrics in `lora_metrics.json` next to this card

Exit criteria: within ~1-2 pts of the full fine-tune accuracy
(`artifacts/baseline/baseline_metrics.json`).

## Intended use & limitations

Research baseline only -- the clean (un-poisoned) LoRA reference for the
FedKnob partition-heterogeneity experiments. Not for production use.
