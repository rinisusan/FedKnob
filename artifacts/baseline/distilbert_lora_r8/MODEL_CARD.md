# Model Card -- FedKnob LoRA baseline (DistilBERT / MASSIVE)

## Summary

Rank-8 LoRA fine-tune of `distilbert-base-uncased` for 60-way intent classification
on Amazon MASSIVE (en). Parameter-efficient counterpart of the full
fine-tune; the persistent per-household artifact in the federated phases is the
LoRA adapter saved here.

- **Base model:** distilbert-base-uncased (66M params, frozen)
- **Task:** sequence classification (60 MASSIVE intents)
- **Method:** PEFT LoRA -- r=8, alpha=16, dropout=0.05,
  target_modules=['q_lin', 'v_lin'], modules_to_save=['pre_classifier', 'classifier', 'classifier', 'score']
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
persist on-device per household. The classification head must also train
because `from_pretrained(num_labels=...)` initialises it randomly; it belongs
to the shared global model, not the per-household adapter.

## Training

- Learning rate: 0.0002 (LoRA; 10x the full-fine-tune lr)
- Epochs: 10.0, batch size 32, max sequence length 64
- Optimizer/schedule: HF `Trainer` defaults (AdamW); fp16 when CUDA available
- Seed: 42

Reproduce with:

```bash
python scripts/train_lora_massive.py
```

## Evaluation (MASSIVE test split)

- accuracy: 0.8806
- full metrics in `lora_metrics.json` next to this card

Exit criteria: within ~1-2 pts of the full fine-tune accuracy
(`artifacts/baseline/baseline_metrics.json`).

## Intended use & limitations

Research baseline only -- the centralized LoRA reference for the
FedKnob partition-heterogeneity experiments. Not for production use.
