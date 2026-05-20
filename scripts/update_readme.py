#!/usr/bin/env python3
"""Update README.md with actual training metrics from Kaggle output.

Reads metrics.json from the Kaggle kernel output directory and replaces
placeholder values (0.XX) in the README results table.
"""

import json
import re
import sys
from pathlib import Path


def load_metrics(metrics_path: str = "kaggle_output/metrics.json") -> dict:
    """Load metrics from the Kaggle output directory."""
    path = Path(metrics_path)
    if not path.exists():
        print(f"ERROR: Metrics file not found at {path}")
        print("Make sure the Kaggle kernel has completed and output was downloaded.")
        sys.exit(1)

    with open(path) as f:
        metrics = json.load(f)

    print(f"Loaded metrics: {json.dumps(metrics, indent=2)}")
    return metrics


def update_readme(readme_path: str = "README.md", metrics: dict = None) -> None:
    """Replace placeholder scores in README.md with actual metrics."""
    path = Path(readme_path)
    if not path.exists():
        print(f"ERROR: README not found at {path}")
        sys.exit(1)

    content = path.read_text(encoding="utf-8")
    original = content

    accuracy = f"{metrics.get('accuracy', 0.0):.4f}"
    f1_score = f"{metrics.get('f1', 0.0):.4f}"
    eval_loss = f"{metrics.get('eval_loss', 0.0):.4f}"

    # Replace in the results table - handle both 0.XX placeholders and previous values
    content = re.sub(
        r'(\|\s*Accuracy\s*\|)\s*[\d.X]+\s*(\|)',
        f'\\1 {accuracy} \\2',
        content
    )
    content = re.sub(
        r'(\|\s*F1 Score\s*\|)\s*[\d.X]+\s*(\|)',
        f'\\1 {f1_score} \\2',
        content
    )
    content = re.sub(
        r'(\|\s*Eval Loss\s*\|)\s*[\d.X]+\s*(\|)',
        f'\\1 {eval_loss} \\2',
        content
    )

    if content != original:
        path.write_text(content, encoding="utf-8")
        print(f"README.md updated with metrics:")
        print(f"  Accuracy:  {accuracy}")
        print(f"  F1 Score:  {f1_score}")
        print(f"  Eval Loss: {eval_loss}")
    else:
        print("WARNING: No placeholders were replaced. Check README format.")


def main():
    metrics = load_metrics()
    update_readme(metrics=metrics)
    print("Done!")


if __name__ == "__main__":
    main()
