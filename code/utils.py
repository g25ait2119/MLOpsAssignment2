"""Shared helpers used across data, train, and eval scripts."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from sklearn.metrics import accuracy_score, f1_score


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 512
CACHED_MODEL_DIR = "distilbert-reviews-genres"
DATA_CACHE_PATH = "genre_reviews_dict.pickle"
ENCODED_DATA_PATH = "encoded_data.pt"
RESULTS_DIR = "./results"
LOGS_DIR = "./logs"

# Use GPU when available; fall back to CPU otherwise so scripts run anywhere.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GENRE_URL_DICT: Dict[str, str] = {
    "poetry":                 "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "children":               "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz",
    "comics_graphic":         "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "fantasy_paranormal":     "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "history_biography":      "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
    "mystery_thriller_crime": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "romance":                "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "young_adult":            "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}


# ---------------------------------------------------------------------------
# Custom torch Dataset wrapping HuggingFace tokenizer encodings
# ---------------------------------------------------------------------------
class MyDataset(torch.utils.data.Dataset):
    """Wraps tokenizer encodings + integer labels for use with Trainer."""

    def __init__(self, encodings, labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------
def build_label_maps(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build label2id and id2label dictionaries from a list of string labels.

    Sorted to keep mapping deterministic across runs.
    """
    unique_labels = sorted(set(labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(pred):
    """HuggingFace Trainer-compatible metric function returning accuracy and F1."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }
