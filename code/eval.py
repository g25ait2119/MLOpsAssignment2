"""Evaluate the fine-tuned DistilBERT model on the held-out test set.

Run from the command line (after `python train.py`):
    python eval.py
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from sklearn.metrics import classification_report
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

from data import load_encoded
from utils import (
    CACHED_MODEL_DIR,
    DEVICE,
    MyDataset,
    compute_metrics,
)


def build_eval_trainer(model_dir: str, eval_batch_size: int) -> Trainer:
    """Load a fine-tuned model from disk and wrap it in a Trainer for prediction."""
    model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(DEVICE)

    eval_args = TrainingArguments(
        output_dir="./eval_tmp",
        per_device_eval_batch_size=eval_batch_size,
        report_to=["wandb"],
    )

    return Trainer(model=model, args=eval_args, compute_metrics=compute_metrics)


def predict_labels(
    trainer: Trainer,
    test_dataset: MyDataset,
    id2label: Dict[int, str],
) -> List[str]:
    """Run prediction and convert the highest-probability ids back to genre strings."""
    predicted_results = trainer.predict(test_dataset)
    predicted_ids = predicted_results.predictions.argmax(-1).flatten().tolist()
    return [id2label[i] for i in predicted_ids]


def save_results(
    results_dir: str,
    metrics: Dict,
    report_text: str,
    report_dict: Dict,
    test_labels: List[str],
    predicted_labels: List[str],
) -> None:
    """Persist evaluation metrics, classification report, and per-example predictions."""
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(report_text)

    with open(os.path.join(results_dir, "classification_report.json"), "w") as f:
        json.dump(report_dict, f, indent=2)

    with open(os.path.join(results_dir, "predictions.json"), "w") as f:
        json.dump(
            [
                {"true_label": t, "predicted_label": p}
                for t, p in zip(test_labels, predicted_labels)
            ],
            f,
            indent=2,
        )

    print(f"Saved evaluation results to {results_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned classifier.")
    parser.add_argument("--model-dir", type=str, default=CACHED_MODEL_DIR,
                        help="Directory containing the fine-tuned model.")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--results-dir", type=str, default="eval_results",
                        help="Where to write evaluation artefacts.")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    payload = load_encoded()
    test_dataset = MyDataset(payload["test_encodings"], payload["test_labels_encoded"])
    id2label = payload["id2label"]
    test_labels = payload["test_labels"]

    import wandb
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "mlops-assignment2"),
        name="distilbert-eval-1",
        job_type="evaluation"
    )

    trainer = build_eval_trainer(args.model_dir, args.eval_batch_size)

    # Trainer.evaluate() reports loss + our compute_metrics dict.
    metrics = trainer.evaluate(eval_dataset=test_dataset)
    print("Evaluation metrics:", metrics)

    # Log final metrics to W&B explicitly
    wandb.log({
        "final/loss": metrics.get("eval_loss"),
        "final/accuracy": metrics.get("eval_accuracy"),
        "final/f1": metrics.get("eval_f1", 0.0), # Assuming f1 might be added, or just fallbacks
    })

    predicted_labels = predict_labels(trainer, test_dataset, id2label)

    report_text = classification_report(test_labels, predicted_labels)
    report_dict = classification_report(test_labels, predicted_labels, output_dict=True)
    print(report_text)

    save_results(
        results_dir=args.results_dir,
        metrics=metrics,
        report_text=report_text,
        report_dict=report_dict,
        test_labels=test_labels,
        predicted_labels=predicted_labels,
    )

    # Upload to W&B as a versioned Artifact
    artifact = wandb.Artifact("eval-report", type="evaluation")
    artifact.add_file(os.path.join(args.results_dir, "classification_report.json"))
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
