"""DistilBERT sequence-classifier builder for the baseline.

Plain full fine-tune -- every parameter trainable, no LoRA. The parameter-
efficient LoRA variant used in the later federated phases lives alongside this
module and should roughly match its accuracy while training <1% of the weights.
"""

from __future__ import annotations

MODEL_NAME = "distilbert-base-uncased"


def build_model(num_labels: int):
    """Return a full DistilBERT sequence classifier (all params trainable)."""
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
