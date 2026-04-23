#!/usr/bin/env python3
"""
Shared reducer for all four diabetes prediction models.

Receives tab-separated lines of the form:
    1\tpred,actual
where pred and actual are integer class labels (0 or 1).

Outputs accuracy, precision, recall, F1, and a confusion matrix.
All mappers emit key=1 so a single reducer gets the full dataset.
"""
import sys

tp = tn = fp = fn = 0

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    parts = line.split('\t')
    if len(parts) != 2:
        continue
    try:
        pred, actual = int(parts[0]), int(parts[1])
    except ValueError:
        continue

    if pred == 1 and actual == 1:
        tp += 1
    elif pred == 0 and actual == 0:
        tn += 1
    elif pred == 1 and actual == 0:
        fp += 1
    elif pred == 0 and actual == 1:
        fn += 1

total = tp + tn + fp + fn
if total == 0:
    print("No predictions received.")
    sys.exit(0)

accuracy  = (tp + tn) / total
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1        = (2 * precision * recall / (precision + recall)
             if (precision + recall) > 0 else 0.0)

print("=== Evaluation Results ===")
print(f"Total samples : {total}")
print(f"Accuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")
print("")
print("Confusion Matrix:")
print(f"               Pred=0   Pred=1")
print(f"  Actual=0     {tn:6d}   {fp:6d}")
print(f"  Actual=1     {fn:6d}   {tp:6d}")
