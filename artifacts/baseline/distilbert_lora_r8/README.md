---
base_model: distilbert-base-uncased
library_name: peft
license: apache-2.0
language:
- en
datasets:
- AmazonScience/massive
metrics:
- accuracy
- f1
pipeline_tag: text-classification
tags:
- base_model:adapter:distilbert-base-uncased
- lora
- transformers
---

# distilbert_lora_r8 — FedKnob Week 2 LoRA baseline

Rank-8 LoRA adapter for `distilbert-base-uncased`, fine-tuned for 60-way intent
classification on Amazon MASSIVE (en-US). This is the parameter-efficient
counterpart of the FedKnob Week 1 full fine-tune and the clean (un-poisoned)
reference adapter for the project's backdoor-contagion experiments: it is the
per-client adapter recipe the federated phases persist across rounds.

## Model Details

### Model Description

The adapter trains only the LoRA A/B matrices on the attention query/value
projections (`q_lin`, `v_lin`) plus the classification head, leaving the 66M
DistilBERT backbone frozen. The persistent per-household artifact — the LoRA
matrices proper — is 147,456 parameters (~0.22% of the model), the property
that makes per-household persistent adapters feasible in federated deployment.

- **Developed by:** FedKnob project 
- **Model type:** PEFT LoRA adapter for DistilBERT sequence classification (60 intents)
- **Language(s) (NLP):** English (en-US)
- **License:** Apache-2.0
- **Finetuned from model:** distilbert-base-uncased


## Training Details

### Training Data

Amazon MASSIVE 1.0, en-US locale (official `alexa/massive` JSONL release),
loaded via `fedknob.data.massive` — the same single-source loader used by
the Week 1 baseline and the federated `worker_id` partitioning. Train split:
~11.5K utterances, 60 intents.

### Training Procedure

#### Preprocessing

DistilBERT tokenizer, truncation at max_length=64; tokenized datasets cached to
disk under `artifacts/tokenized/`. Stable sorted intent-name → id label map
shared with the Week 1 baseline.

#### Training Hyperparameters

- **LoRA:** r=8, alpha=16, dropout=0.05, target_modules=["q_lin", "v_lin"],
  modules_to_save=["pre_classifier", "classifier", "score"], bias="none"
- **Learning rate:** 2e-4 (10x the full-fine-tune lr)
- **Epochs:** 10; **batch size:** 32 (train) / 64 (eval)
- **Optimizer/schedule:** HF `Trainer` defaults (AdamW, linear decay)
- **Training regime:** fp16 mixed precision (CUDA)
- **Seed:** 42

#### Speeds, Sizes, Times

- Trainable parameters: ~784K (~1.2% incl. classification head copies)
- LoRA adapter proper: 147,456 params (~0.22%) — the persistent per-household bytes
- Adapter checkpoint (`adapter_model.safetensors`): ~3 MB

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

MASSIVE en-US test split (2,974 utterances).

#### Metrics

Accuracy (headline), balanced accuracy and macro precision/recall/F1 (robust to
MASSIVE's class imbalance — a backdoored adapter can hide behind plain
accuracy), weighted F1, and top-5 accuracy.

### Results

| metric | value |
|---|---|
| eval_loss | 0.4478 |
| accuracy | 0.8806 |
| balanced_accuracy | 0.8711 |
| f1_macro | 0.8581 |
| f1_weighted | 0.8807 |
| precision_macro | 0.8589 |
| recall_macro | 0.8711 |
| top5_accuracy | 0.9805 |

#### Summary

At 88.06% accuracy, within ~0.64 pts of the Week 1 full fine-tune (88.70%
accuracy) while training <1% of parameters in the persistent adapter — **Week 2
exit criteria: PASS** (target: within ~1–2 pts). Full metrics in
`lora_metrics.json`.

## Technical Specifications

### Model Architecture and Objective

DistilBERT-base (6 layers, 768 hidden, 66M params, frozen) + rank-8 LoRA on
attention q/v projections + 60-way classification head; single-label
cross-entropy objective.

### Compute Infrastructure

#### Hardware

NVIDIA RTX 5060 (Blackwell, sm_120), 8 GB VRAM, single GPU.

#### Software

PyTorch (nightly cu128), HuggingFace Transformers 5.x, Datasets 4.x, PEFT.
### Framework versions

- PEFT 0.19.1