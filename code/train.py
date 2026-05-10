"""Fine-tune DistilBERT on the encoded Goodreads genre dataset.

Run from the command line (after `python data.py`):
    python train.py
"""

from __future__ import annotations

import argparse
import os

from transformers import (
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from data import load_encoded
from utils import (
    CACHED_MODEL_DIR,
    DEVICE,
    LOGS_DIR,
    MODEL_NAME,
    MyDataset,
    RESULTS_DIR,
    compute_metrics,
)

# Disable Weights & Biases logging which would otherwise prompt for an API key.
os.environ["WANDB_DISABLED"] = "true"


def build_trainer(
    train_dataset: MyDataset,
    test_dataset: MyDataset,
    num_labels: int,
    num_train_epochs: int,
    train_batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    warmup_steps: int,
    weight_decay: float,
    logging_steps: int,
    output_dir: str,
    logging_dir: str,
) -> Trainer:
    """Construct the DistilBERT model and wrap it in a HuggingFace Trainer."""
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    ).to(DEVICE)

    training_args = TrainingArguments(
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        output_dir=output_dir,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        eval_strategy="steps",
        report_to=[],
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for genre classification.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--logging-dir", type=str, default=LOGS_DIR)
    parser.add_argument("--model-out", type=str, default=CACHED_MODEL_DIR,
                        help="Directory to save the fine-tuned model.")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    payload = load_encoded()
    train_dataset = MyDataset(payload["train_encodings"], payload["train_labels_encoded"])
    test_dataset = MyDataset(payload["test_encodings"], payload["test_labels_encoded"])
    num_labels = len(payload["id2label"])

    trainer = build_trainer(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_labels=num_labels,
        num_train_epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
    )

    trainer.train()

    trainer.save_model(args.model_out)
    print(f"Saved fine-tuned model to {args.model_out}")


if __name__ == "__main__":
    main()
