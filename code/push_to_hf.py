"""Push the trained model and tokenizer to Hugging Face Hub.

This script is designed to be run in CI/CD (GitHub Actions) or locally.
It loads the fine-tuned model from a local directory (if available) or
falls back to the base pre-trained model, then pushes both model and
tokenizer to the specified Hugging Face Hub repository.

Usage:
    python push_to_hf.py

Environment variables:
    HF_TOKEN    - Hugging Face API token with write access
    HF_REPO_ID  - Target repo, e.g. 'sureshbabugandla/ML_OPS_ASSIGNMENT2'
"""

from __future__ import annotations

import os
import sys

from huggingface_hub import login as hf_login
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

from utils import CACHED_MODEL_DIR, MODEL_NAME

# Number of genre labels used in training
NUM_LABELS = 8


def main() -> None:
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo = os.environ.get("HF_REPO_ID")

    if not hf_repo:
        print("ERROR: HF_REPO_ID environment variable is not set.")
        sys.exit(1)

    if not hf_token:
        print("ERROR: HF_TOKEN environment variable is not set.")
        sys.exit(1)

    # Authenticate with Hugging Face Hub
    print("Logging in to Hugging Face Hub...")
    hf_login(token=hf_token)

    # Try to load the fine-tuned model; fall back to base pre-trained model
    if os.path.isdir(CACHED_MODEL_DIR):
        print(f"Loading fine-tuned model from {CACHED_MODEL_DIR}...")
        model = DistilBertForSequenceClassification.from_pretrained(CACHED_MODEL_DIR)
    else:
        print(f"Fine-tuned model not found at {CACHED_MODEL_DIR}.")
        print(f"Loading base pre-trained model ({MODEL_NAME}) with {NUM_LABELS} labels...")
        model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS
        )

    # Load tokenizer
    print(f"Loading tokenizer ({MODEL_NAME})...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    # Push to Hub
    print(f"Pushing model to HF Hub: {hf_repo}")
    model.push_to_hub(hf_repo)

    print(f"Pushing tokenizer to HF Hub: {hf_repo}")
    tokenizer.push_to_hub(hf_repo)

    print(f"Successfully pushed model and tokenizer to https://huggingface.co/{hf_repo}")


if __name__ == "__main__":
    main()
