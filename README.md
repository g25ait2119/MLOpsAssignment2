# MLOps Assignment 2: Hugging Face Fine-Tuning & Deployment

This repository contains the complete MLOps workflow for fine-tuning a DistilBERT model on the Goodreads genre classification dataset. The workflow translates a standard Jupyter notebook into production-ready Python scripts, tracks experiments using Weights & Biases (W&B), and automatically publishes the trained model to the Hugging Face Hub via GitHub Actions.

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

## Results

| Metric | Score  |
|-----------|--------|
| Accuracy  | 0.XX   |
| F1 Score  | 0.XX   |
| Eval Loss | 0.XX   |

*(Note: Replace `0.XX` with the final metrics once training is completely finished).*

## Important Links

- **Hugging Face Model:** [https://huggingface.co/sureshbabugandla/ML_OPS_ASSIGNMENT2](https://huggingface.co/sureshbabugandla/ML_OPS_ASSIGNMENT2)
- **W&B Dashboard:** [https://wandb.ai/sureshbabugandla/mlops-assignment2](https://wandb.ai/sureshbabugandla/mlops-assignment2)
