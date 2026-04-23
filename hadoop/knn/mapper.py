#!/usr/bin/env python3
"""
Mapper for K-Nearest Neighbors.

K-Nearest_Neighbors.pkl is ~100 MB because sklearn stores the full training
set inside the model (required for distance computation at predict time).
Memory usage per mapper task is therefore ~100 MB model + per-row overhead,
which is manageable on most cluster nodes.

CSV column layout (0-indexed):
  0        : DIABETE3  (target label, 0 or 1)
  1 - 16   : features  BPHIGH4, TOLDHI2, CVDINFR4, CVDSTRK3, _BMI5,
                       GENHLTH, SMOKE100, EXERANY2, ALCDAY5, SEX,
                       _AGE_G, _RACE, EDUCA, INCOME2, HLTHPLN1, year
"""
import sys
import pickle
import numpy as np

MODEL_FILE = "K-Nearest_Neighbors.pkl"
FEATURE_INDICES = list(range(1, 17))

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# KNN predict_proba / predict on batches is faster than row-by-row.
# Buffer rows and flush every BATCH_SIZE to balance latency and throughput.
BATCH_SIZE = 512
actuals  = []
features_buf = []

def flush():
    if not features_buf:
        return
    X = np.array(features_buf, dtype=np.float64)
    preds = model.predict(X)
    for pred, actual in zip(preds, actuals):
        print(f"{int(pred)}\t{actual}")
    actuals.clear()
    features_buf.clear()

for line in sys.stdin:
    line = line.strip()
    if not line or line.startswith("DIABETE3"):
        continue
    parts = line.split(",")
    if len(parts) < 17:
        continue
    try:
        actual = int(float(parts[0]))
        features = [float(parts[i]) for i in FEATURE_INDICES]
        actuals.append(actual)
        features_buf.append(features)
        if len(features_buf) >= BATCH_SIZE:
            flush()
    except (ValueError, IndexError):
        continue

flush()
