"""Data pipeline for the Goodreads genre classification task.

Responsibilities:
    * Download streaming review data per genre from the UCSD Goodreads dataset.
    * Randomly sample a manageable subset of reviews per genre.
    * Split into train / test sets.
    * Tokenize using the DistilBERT tokenizer.
    * Persist the encoded train / test datasets for use by train.py and eval.py.

Run from the command line:
    python data.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import random
from typing import Dict, List, Tuple

import requests
import torch
from transformers import DistilBertTokenizerFast

from utils import (
    DATA_CACHE_PATH,
    ENCODED_DATA_PATH,
    GENRE_URL_DICT,
    MAX_LENGTH,
    MODEL_NAME,
    build_label_maps,
)


# ---------------------------------------------------------------------------
# Download / sample
# ---------------------------------------------------------------------------
def load_reviews(url: str, head: int = 10000, sample_size: int = 2000) -> List[str]:
    """Stream a gzipped JSON-lines review file from `url` and return a random sample.

    Reads up to `head` reviews, then samples `sample_size` of them.
    """
    reviews: List[str] = []
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with gzip.open(response.raw, "rt", encoding="utf-8") as file:
        for count, line in enumerate(file, start=1):
            d = json.loads(line)
            reviews.append(d["review_text"])
            if head is not None and count >= head:
                break

    return random.sample(reviews, min(sample_size, len(reviews)))


def download_all_genres(
    head: int = 10000,
    sample_size: int = 2000,
    cache_path: str = DATA_CACHE_PATH,
) -> Dict[str, List[str]]:
    """Download and sample reviews for every genre, caching to a pickle file."""
    genre_reviews_dict: Dict[str, List[str]] = {}
    for genre, url in GENRE_URL_DICT.items():
        print(f"Loading reviews for genre: {genre}")
        genre_reviews_dict[genre] = load_reviews(url, head=head, sample_size=sample_size)

    with open(cache_path, "wb") as f:
        pickle.dump(genre_reviews_dict, f)
    print(f"Cached raw reviews to {cache_path}")
    return genre_reviews_dict


def load_or_download(
    head: int = 10000,
    sample_size: int = 2000,
    cache_path: str = DATA_CACHE_PATH,
) -> Dict[str, List[str]]:
    """Load cached reviews if present; otherwise download and cache them."""
    try:
        with open(cache_path, "rb") as f:
            print(f"Loading cached reviews from {cache_path}")
            return pickle.load(f)
    except FileNotFoundError:
        return download_all_genres(head=head, sample_size=sample_size, cache_path=cache_path)


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
def train_test_split(
    genre_reviews_dict: Dict[str, List[str]],
    per_genre: int = 1000,
    train_size: int = 800,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Sample `per_genre` reviews per genre and split into train / test."""
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

    return train_texts, train_labels, test_texts, test_labels


# ---------------------------------------------------------------------------
# Tokenization / encoding
# ---------------------------------------------------------------------------
def encode_data(
    train_texts: List[str],
    train_labels: List[str],
    test_texts: List[str],
    test_labels: List[str],
    model_name: str = MODEL_NAME,
    max_length: int = MAX_LENGTH,
):
    """Tokenize texts and encode string labels to ints. Returns encodings + maps."""
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    label2id, id2label = build_label_maps(train_labels)

    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, max_length=max_length
    )
    test_encodings = tokenizer(
        test_texts, truncation=True, padding=True, max_length=max_length
    )

    train_labels_encoded = [label2id[y] for y in train_labels]
    test_labels_encoded = [label2id[y] for y in test_labels]

    return (
        train_encodings,
        train_labels_encoded,
        test_encodings,
        test_labels_encoded,
        label2id,
        id2label,
    )


def save_encoded(
    train_encodings,
    train_labels_encoded: List[int],
    test_encodings,
    test_labels_encoded: List[int],
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    test_texts: List[str],
    test_labels: List[str],
    path: str = ENCODED_DATA_PATH,
) -> None:
    """Persist the encoded train / test bundle so train.py and eval.py can reuse it."""
    payload = {
        "train_encodings": dict(train_encodings),
        "train_labels_encoded": train_labels_encoded,
        "test_encodings": dict(test_encodings),
        "test_labels_encoded": test_labels_encoded,
        "label2id": label2id,
        "id2label": id2label,
        "test_texts": test_texts,
        "test_labels": test_labels,
    }
    torch.save(payload, path)
    print(f"Saved encoded data bundle to {path}")


def load_encoded(path: str = ENCODED_DATA_PATH) -> dict:
    """Load the encoded train / test bundle saved by `save_encoded`."""
    return torch.load(path, weights_only=False)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Goodreads genre dataset.")
    parser.add_argument("--head", type=int, default=10000,
                        help="Max reviews to read per genre before sampling.")
    parser.add_argument("--sample-size", type=int, default=2000,
                        help="Random reviews kept per genre.")
    parser.add_argument("--per-genre", type=int, default=1000,
                        help="Reviews per genre used in the train+test split.")
    parser.add_argument("--train-size", type=int, default=800,
                        help="Reviews per genre assigned to the training set.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling / splitting.")
    args = parser.parse_args()

    random.seed(args.seed)

    genre_reviews_dict = load_or_download(head=args.head, sample_size=args.sample_size)
    train_texts, train_labels, test_texts, test_labels = train_test_split(
        genre_reviews_dict, per_genre=args.per_genre, train_size=args.train_size
    )

    print(
        f"Train: {len(train_texts)} texts / {len(train_labels)} labels | "
        f"Test: {len(test_texts)} texts / {len(test_labels)} labels"
    )

    (
        train_encodings,
        train_labels_encoded,
        test_encodings,
        test_labels_encoded,
        label2id,
        id2label,
    ) = encode_data(train_texts, train_labels, test_texts, test_labels)

    save_encoded(
        train_encodings,
        train_labels_encoded,
        test_encodings,
        test_labels_encoded,
        label2id,
        id2label,
        test_texts,
        test_labels,
    )


if __name__ == "__main__":
    main()
