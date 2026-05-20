"""
Self-contained Kaggle Kernel — MLOps Assignment 2 Training Pipeline
===================================================================

This script is designed to run as a single-file Kaggle kernel.
It cannot import from local project modules (data.py, train.py, utils.py, etc.)
because Kaggle kernels execute only a single file. All logic is therefore inlined.

Pipeline steps:
  1. Install missing packages
  2. Load Kaggle secrets (WANDB_API_KEY, HF_TOKEN)
  3. Download & cache Goodreads review data (8 genres)
  4. Tokenize and encode data
  5. Fine-tune DistilBERT for sequence classification
  6. Evaluate and produce a classification report
  7. Log metrics & artifacts to Weights & Biases
  8. Push the trained model + tokenizer to HuggingFace Hub
  9. Save metrics.json for downstream GitHub Actions consumption
"""

# ── Step 0: Install missing packages ─────────────────────────────────────────
import subprocess, sys
subprocess.check_call(
    [sys.executable, '-m', 'pip', 'install', '-q',
     'wandb', 'huggingface_hub', 'accelerate>=1.1.0']
)

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import json
import gzip
import random
import pickle

import numpy as np
import torch
import requests
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ── Step 1: Load Kaggle Secrets ──────────────────────────────────────────────
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    WANDB_API_KEY = secrets.get_secret('WANDB_API_KEY')
    HF_TOKEN = secrets.get_secret('HF_TOKEN')
    print('Loaded secrets from Kaggle vault.')
except Exception:
    # Fallback for non-Kaggle environments
    WANDB_API_KEY = os.environ.get('WANDB_API_KEY', '')
    HF_TOKEN = os.environ.get('HF_TOKEN', '')
    print('Using environment variables for secrets (non-Kaggle fallback).')

os.environ['WANDB_API_KEY'] = WANDB_API_KEY
os.environ['HF_TOKEN'] = HF_TOKEN

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_NAME = 'distilbert-base-cased'
MAX_LENGTH = 512
CACHED_MODEL_DIR = 'distilbert-reviews-genres'
DATA_CACHE_PATH = 'genre_reviews_dict.pickle'
ENCODED_DATA_PATH = 'encoded_data.pt'
RESULTS_DIR = './results'
LOGS_DIR = './logs'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HF_REPO_ID = 'sureshbabugandla/ML_OPS_ASSIGNMENT2'

GENRE_URL_DICT = {
    'poetry': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz',
    'children': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz',
    'comics_graphic': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz',
    'fantasy_paranormal': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz',
    'history_biography': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz',
    'mystery_thriller_crime': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz',
    'romance': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz',
    'young_adult': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz',
}


# ── Dataset wrapper ──────────────────────────────────────────────────────────
class MyDataset(torch.utils.data.Dataset):
    """Simple map-style dataset wrapping HuggingFace tokenizer encodings."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ── Helper functions ─────────────────────────────────────────────────────────

def build_label_maps(labels):
    """Create label ↔ id mappings from a list of string labels."""
    unique = sorted(set(labels))
    label2id = {lbl: idx for idx, lbl in enumerate(unique)}
    id2label = {idx: lbl for idx, lbl in enumerate(unique)}
    return label2id, id2label


def compute_metrics(pred):
    """Compute accuracy and weighted F1 for the HF Trainer."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1}


def load_reviews(url, head=10000, sample_size=2000):
    """Stream a gzipped JSON-lines file and return a random sample of review texts."""
    print(f'  Downloading: {url}')
    reviews = []
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with gzip.open(response.raw, 'rt', encoding='utf-8') as f:
        for count, line in enumerate(f, start=1):
            d = json.loads(line)
            reviews.append(d['review_text'])
            if head is not None and count >= head:
                break
    sampled = random.sample(reviews, min(sample_size, len(reviews)))
    print(f'    Sampled {len(sampled)} reviews.')
    return sampled


def download_all_genres(head=10000, sample_size=2000, cache_path=DATA_CACHE_PATH):
    """Download reviews for every genre and cache the result."""
    genre_reviews_dict = {}
    for genre, url in GENRE_URL_DICT.items():
        print(f'Processing genre: {genre}')
        genre_reviews_dict[genre] = load_reviews(url, head=head, sample_size=sample_size)
    with open(cache_path, 'wb') as f:
        pickle.dump(genre_reviews_dict, f)
    print(f'Cached genre reviews to {cache_path}')
    return genre_reviews_dict


def load_or_download(head=10000, sample_size=2000, cache_path=DATA_CACHE_PATH):
    """Return cached data if available; otherwise download fresh."""
    if os.path.exists(cache_path):
        print(f'Loading cached data from {cache_path}')
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return download_all_genres(head=head, sample_size=sample_size, cache_path=cache_path)


def train_test_split(genre_reviews_dict, per_genre=1000, train_size=800):
    """Sample per_genre reviews per genre and split into train / test."""
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []
    for genre, reviews in genre_reviews_dict.items():
        sampled = random.sample(reviews, min(per_genre, len(reviews)))
        for review in sampled[:train_size]:
            train_texts.append(review)
            train_labels.append(genre)
        for review in sampled[train_size:]:
            test_texts.append(review)
            test_labels.append(genre)
    print(f'Train size: {len(train_texts)}, Test size: {len(test_texts)}')
    return train_texts, train_labels, test_texts, test_labels


def encode_data(train_texts, train_labels, test_texts, test_labels,
                model_name=MODEL_NAME, max_length=MAX_LENGTH):
    """Tokenize texts and encode string labels to integer ids."""
    print('Tokenizing data …')
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

    label2id, id2label = build_label_maps(train_labels)
    train_labels_encoded = [label2id[lbl] for lbl in train_labels]
    test_labels_encoded = [label2id[lbl] for lbl in test_labels]

    print(f'Tokenization complete. Labels: {list(id2label.values())}')
    return train_encodings, train_labels_encoded, test_encodings, test_labels_encoded, label2id, id2label


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    print(f'Using device: {DEVICE}')

    # ── Step 1: Data ─────────────────────────────────────────────────────────
    print('\n=== Step 1: Loading data ===')
    genre_reviews_dict = load_or_download()
    train_texts, train_labels, test_texts, test_labels = train_test_split(genre_reviews_dict)

    (train_encodings, train_labels_encoded,
     test_encodings, test_labels_encoded,
     label2id, id2label) = encode_data(train_texts, train_labels, test_texts, test_labels)

    train_dataset = MyDataset(train_encodings, train_labels_encoded)
    test_dataset = MyDataset(test_encodings, test_labels_encoded)
    num_labels = len(id2label)
    print(f'Number of labels: {num_labels}')

    # ── Step 2: Weights & Biases init ────────────────────────────────────────
    print('\n=== Step 2: Initialising W&B ===')
    import wandb
    wandb.init(
        project='mlops-assignment2',
        name='distilbert-run-1',
        config={
            'model': MODEL_NAME,
            'epochs': 3,
            'batch_size': 16,
            'learning_rate': 3e-5,
            'max_length': MAX_LENGTH,
            'dataset': 'UCSD Goodreads',
            'platform': 'Kaggle',
        },
    )

    # ── Step 3: Model & Training ─────────────────────────────────────────────
    print('\n=== Step 3: Training ===')
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels,
    ).to(DEVICE)

    training_args = TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        warmup_steps=100,
        weight_decay=0.01,
        output_dir=RESULTS_DIR,
        logging_dir=LOGS_DIR,
        logging_steps=50,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        report_to=['wandb'],
        run_name='distilbert-run-1',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(CACHED_MODEL_DIR)
    print(f'Model saved to {CACHED_MODEL_DIR}')

    # ── Step 4: Evaluation ───────────────────────────────────────────────────
    print('\n=== Step 4: Evaluation ===')
    eval_results = trainer.evaluate(eval_dataset=test_dataset)
    print('Evaluation metrics:', eval_results)

    wandb.log({
        'final/loss': eval_results.get('eval_loss'),
        'final/accuracy': eval_results.get('eval_accuracy'),
        'final/f1': eval_results.get('eval_f1', 0.0),
    })

    # Predictions & classification report
    predicted_results = trainer.predict(test_dataset)
    predicted_ids = predicted_results.predictions.argmax(-1).flatten().tolist()
    predicted_labels = [id2label[i] for i in predicted_ids]

    report_dict = classification_report(
        test_labels, predicted_labels,
        target_names=list(id2label.values()), output_dict=True,
    )
    report_text = classification_report(
        test_labels, predicted_labels,
        target_names=list(id2label.values()),
    )
    print(report_text)

    # Save eval_report.json
    with open('eval_report.json', 'w') as f:
        json.dump(report_dict, f, indent=2)
    print('Saved eval_report.json')

    # Upload to W&B as artifact
    artifact = wandb.Artifact('eval-report', type='evaluation')
    artifact.add_file('eval_report.json')
    wandb.log_artifact(artifact)
    print('Uploaded eval-report artifact to W&B')

    # ── Step 5: Push to HuggingFace Hub ──────────────────────────────────────
    print('\n=== Step 5: HuggingFace Hub ===')
    if HF_TOKEN:
        from huggingface_hub import login as hf_login
        hf_login(token=HF_TOKEN)
        trainer.model.push_to_hub(HF_REPO_ID)
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
        tokenizer.push_to_hub(HF_REPO_ID)
        print(f'Pushed model and tokenizer to https://huggingface.co/{HF_REPO_ID}')
        wandb.run.summary['huggingface_model'] = f'https://huggingface.co/{HF_REPO_ID}'
    else:
        print('HF_TOKEN not set — skipping HuggingFace Hub push.')

    # ── Step 6: Save metrics for GitHub Actions ──────────────────────────────
    print('\n=== Step 6: Saving metrics.json ===')
    metrics_output = {
        'accuracy': eval_results.get('eval_accuracy', 0.0),
        'f1': eval_results.get('eval_f1', 0.0),
        'eval_loss': eval_results.get('eval_loss', 0.0),
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics_output, f, indent=2)
    print('Saved metrics.json for downstream consumption')

    wandb.finish()
    print('\n✅ Pipeline complete!')


if __name__ == '__main__':
    main()
