#!/usr/bin/env python3
"""
Hadoop Streaming reducer: aggregate per-row predictions into metrics.

Input  (stdin):  "actual\tpredicted,probability" lines from mapper
Output (stdout): confusion matrix counts + accuracy, precision, recall, F1
"""
import sys

def main():
    tp = tn = fp = fn = 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            actual_str, pred_str = line.split("\t")
            actual = int(actual_str)
            pred = int(pred_str.split(",")[0])
        except ValueError:
            continue

        if actual == 1 and pred == 1:
            tp += 1
        elif actual == 0 and pred == 0:
            tn += 1
        elif actual == 0 and pred == 1:
            fp += 1
        elif actual == 1 and pred == 0:
            fn += 1

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

if __name__ == "__main__":
    main()
