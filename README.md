# MLOps Assignment 2: Hugging Face Fine-Tuning & Deployment

This repository contains the complete MLOps workflow for fine-tuning a DistilBERT model on the Goodreads genre classification dataset. The model is trained on **Kaggle Notebooks** (free GPU), experiments are tracked using Weights & Biases (W&B), and the trained model is published to the Hugging Face Hub.

## Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the data pipeline:**
   This downloads and encodes the Goodreads reviews dataset.
   ```bash
   python code/data.py
   ```

3. **Train the model:**
   This fine-tunes DistilBERT and logs training metrics to W&B. Ensure you have the `WANDB_API_KEY` set.
   ```bash
   export WANDB_API_KEY="your-wandb-api-key"
   python code/train.py
   ```

4. **Evaluate the model:**
   This evaluates the model on the test set and uploads the classification report to W&B.
   ```bash
   python code/eval.py
   ```

## Training Platform

Training was performed on **Kaggle Notebooks** with GPU T4 acceleration. Kaggle Secrets were used for securely storing `WANDB_API_KEY` and `HF_TOKEN`.

## Automated Kaggle Training (CI/CD)

This project includes a **GitHub Actions workflow** that automatically triggers training on Kaggle GPUs.

### How it works:
1. On push to `main` (or manual trigger), the workflow pushes the training kernel to Kaggle
2. The kernel runs on Kaggle's free GPU (T4), training DistilBERT for 3 epochs
3. Metrics are downloaded and the README is auto-updated with real scores
4. The trained model and tokenizer are pushed to Hugging Face Hub

### Required GitHub Secrets:
| Secret | Description |
|--------|-------------|
| `KAGGLE_USERNAME` | Your Kaggle username |
| `KAGGLE_KEY` | Your Kaggle API key (from kaggle.com → Account → API) |
| `WANDB_API_KEY` | Your Weights & Biases API key |
| `HF_API_KEY` | Your Hugging Face token with write access |

### Required Kaggle Secrets:
Configure these in Kaggle (Add-ons → Secrets) and check "Attach to notebook":
- `WANDB_API_KEY` — from wandb.ai/settings
- `HF_TOKEN` — from huggingface.co/settings/tokens

### Trigger training manually:
Go to **Actions** → **Kaggle Training Pipeline** → **Run workflow**

## Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.XX   |
| F1 Score  | 0.XX   |
| Eval Loss | 0.XX   |

*(Scores are auto-updated by the CI/CD pipeline after Kaggle training completes.)*

## Important Links

- **Kaggle Notebook:** [https://www.kaggle.com/sureshbabugandla/mlops-assignment-2-fine-tuning-classification](https://www.kaggle.com/sureshbabugandla/mlops-assignment-2-fine-tuning-classification)
- **Hugging Face Model:** [https://huggingface.co/sureshbabugandla/ML_OPS_ASSIGNMENT2](https://huggingface.co/sureshbabugandla/ML_OPS_ASSIGNMENT2)
- **W&B Dashboard:** [https://wandb.ai/sureshbabugandla/mlops-assignment2](https://wandb.ai/sureshbabugandla/mlops-assignment2)
