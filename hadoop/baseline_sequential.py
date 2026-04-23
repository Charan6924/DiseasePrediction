#!/usr/bin/env python3
"""
Sequential baseline: same inference task as the MapReduce job but single-process.
Run this locally or on one HPCC node to measure wall-clock time for comparison.

Usage:
    python3 baseline_sequential.py <cleaned_data.csv> <model.pkl>
"""
import sys
import time
import warnings
import joblib
import csv

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

FEATURE_COLS = list(range(1, 17))  # cols 1..16 (excludes only DIABETE3 at 0; year at 16 is included)

def load_model(path):
    return joblib.load(path)

def run(data_path, model_path):
    model = load_model(model_path)

    tp = tn = fp = fn = 0
    start = time.perf_counter()

    with open(data_path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # skip header

        for row in reader:
            if len(row) < 16:
                continue
            try:
                actual = int(float(row[0]))
                features = [float(row[i]) for i in FEATURE_COLS]
            except ValueError:
                continue

            pred = int(model.predict([features])[0])

            if actual == 1 and pred == 1:
                tp += 1
            elif actual == 0 and pred == 0:
                tn += 1
            elif actual == 0 and pred == 1:
                fp += 1
            elif actual == 1 and pred == 0:
                fn += 1

    elapsed = time.perf_counter() - start
    total = tp + tn + fp + fn
    accuracy  = (tp + tn) / total              if total > 0             else 0.0
    precision = tp / (tp + fp)                 if (tp + fp) > 0         else 0.0
    recall    = tp / (tp + fn)                 if (tp + fn) > 0         else 0.0
    f1        = (2 * precision * recall
                 / (precision + recall))       if (precision + recall) > 0 else 0.0

    print(f"total_predictions\t{total}")
    print(f"true_positives\t{tp}")
    print(f"true_negatives\t{tn}")
    print(f"false_positives\t{fp}")
    print(f"false_negatives\t{fn}")
    print(f"accuracy\t{accuracy:.4f}")
    print(f"precision\t{precision:.4f}")
    print(f"recall\t{recall:.4f}")
    print(f"f1_score\t{f1:.4f}")
    print(f"wall_time_seconds\t{elapsed:.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 baseline_sequential.py <data.csv> <model.pkl>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
