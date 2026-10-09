# Model Card — FedKnob baseline (DistilBERT / MASSIVE)

## Summary

Full fine-tune of `distilbert-base-uncased` (~66M parameters, no LoRA) for 60-way
intent classification on the Amazon MASSIVE dataset (English). This is the
reference baseline the FedKnob project anchors against.

- **Base model:** distilbert-base-uncased (6 layers, dim 768, 12 heads)
- **Task:** sequence classification (60 MASSIVE intents)
- **Method:** full fine-tuning (all parameters trainable)
- **Dataset:** mteb/amazon_massive_intent, config `en` (parquet mirror of MASSIVE)
- **Language:** English
- **License:** Apache-2.0

## Training

- Optimizer/schedule: HF `Trainer` defaults (AdamW)
- Learning rate: 2e-5 (full fine-tune)
- Epochs: 10, batch size 32, max sequence length 64
- Mixed precision: fp16 when CUDA is available

Reproduce with:

```bash
python scripts/train_baseline_massive.py
```

## Evaluation

Evaluated on the MASSIVE `test` split. Macro metrics are emphasised because
MASSIVE intents are class-imbalanced. Metrics are written to
`baseline_metrics.json` next to this card.

| Metric | Baseline (full FT) |
|---|---|
| Eval loss | 0.4991 |
| Accuracy | 0.8870 |
| Balanced accuracy | 0.8606 |
| F1 (macro) | 0.8505 |
| F1 (weighted) | 0.8863 |
| Precision (macro) | 0.8511 |
| Recall (macro) | 0.8606 |
| Top-5 accuracy | 0.9755 |


Optional diagnostics (`--per-class-report`, `--asr-trigger/--asr-target`)
produce a per-class report, a confusion matrix, and a clean-model Attack Success
Rate control retained from an earlier phase; unused by the measurements here.

[ASR baseline] trigger='cf'  target_id=59  n=2974
  clean-model ASR (control) = 0.0484  (expect near chance ~ 0.0167)

## Intended use & limitations

Research baseline only. The numbers establish a clean reference point for the
project's federated experiments; the model is not intended
for production intent classification.

> Note: `baseline_metrics.json` is regenerated each run. If it predates this
> full-fine-tune baseline (e.g. it was produced by an earlier LoRA experiment),
> rerun the script to refresh it.
