#!/usr/bin/env python3
"""
Read per-model result files written by run_inference.sh and print a
side-by-side comparison table.

Usage:
    python3 hadoop/compare_results.py results/<timestamp>
"""
import sys
from pathlib import Path

METRICS = ["accuracy", "precision", "recall", "f1_score"]

MODEL_ORDER = [
    "Random_Forest",
    "Logistic_Regression",
    "XGBoost",
    "K-Nearest_Neighbors",
]


def parse_result_file(path: Path) -> dict:
    metrics = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if "\t" in line:
            key, value = line.split("\t", 1)
            metrics[key.strip()] = value.strip()
    return metrics


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 compare_results.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.is_dir():
        print(f"ERROR: {results_dir} is not a directory")
        sys.exit(1)

    results = {f.stem: parse_result_file(f) for f in results_dir.glob("*.txt")}

    if not results:
        print("No result files found in", results_dir)
        sys.exit(1)

    ordered = [m for m in MODEL_ORDER if m in results]
    ordered += [m for m in results if m not in MODEL_ORDER]

    col_w, metric_w = 24, 12

    sep = "-" * (col_w + metric_w * len(METRICS))
    header = f"{'Model':<{col_w}}" + "".join(f"{m.capitalize():<{metric_w}}" for m in METRICS)

    print("\n=== Classification Metrics ===")
    print(sep)
    print(header)
    print(sep)
    for model in ordered:
        row = f"{model.replace('_', ' '):<{col_w}}"
        for metric in METRICS:
            row += f"{results[model].get(metric, 'N/A'):<{metric_w}}"
        print(row)
    print(sep)

    cm_keys = ["true_positives", "true_negatives", "false_positives", "false_negatives", "total_predictions"]
    cm_labels = ["TP", "TN", "FP", "FN", "Total"]
    cm_w = 10
    cm_sep = "-" * (col_w + cm_w * len(cm_keys))
    cm_header = f"{'Model':<{col_w}}" + "".join(f"{h:<{cm_w}}" for h in cm_labels)

    print("\n=== Confusion Matrix Counts ===")
    print(cm_sep)
    print(cm_header)
    print(cm_sep)
    for model in ordered:
        row = f"{model.replace('_', ' '):<{col_w}}"
        for key in cm_keys:
            row += f"{results[model].get(key, 'N/A'):<{cm_w}}"
        print(row)
    print(cm_sep)
    print()


if __name__ == "__main__":
    main()
