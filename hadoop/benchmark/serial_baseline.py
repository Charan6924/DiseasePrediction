"""
Serial baseline — runs all four models on the full dataset using a single
process so we can compare wall-clock time against the Hadoop parallel runs.

Usage (from project root):
    python3 hadoop/benchmark/serial_baseline.py

Output: prints a results table to stdout and writes timing_results.txt.
"""
import pickle
import time
import numpy as np
import os

DATA_PATH   = "brfss_diabetes_clean.csv/brfss_diabetes_clean.csv"
MODELS_DIR  = "models"
OUTPUT_FILE = "hadoop/benchmark/timing_results.txt"

MODELS = {
    "Logistic_Regression": "Logistic_Regression.pkl",
    "Random_Forest":       "Random_Forest.pkl",
    "XGBoost":             "XGBoost.pkl",
    "KNN":                 "K-Nearest_Neighbors.pkl",
}

FEATURE_INDICES = list(range(1, 17))


def load_data(path):
    X, y = [], []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i == 0 or not line:
                continue
            parts = line.split(",")
            if len(parts) < 17:
                continue
            try:
                y.append(int(float(parts[0])))
                X.append([float(parts[j]) for j in FEATURE_INDICES])
            except ValueError:
                continue
    return np.array(X, dtype=np.float64), np.array(y, dtype=int)


def evaluate(preds, actuals):
    tp = tn = fp = fn = 0
    for p, a in zip(preds, actuals):
        if   p == 1 and a == 1: tp += 1
        elif p == 0 and a == 0: tn += 1
        elif p == 1 and a == 0: fp += 1
        elif p == 0 and a == 1: fn += 1
    total     = tp + tn + fp + fn
    accuracy  = (tp + tn) / total if total else 0
    precision = tp / (tp + fp)    if (tp + fp) else 0
    recall    = tp / (tp + fn)    if (tp + fn) else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0)
    return dict(accuracy=accuracy, precision=precision,
                recall=recall, f1=f1, total=total)


def predict_xgb(model, X):
    import xgboost as xgb
    if isinstance(model, xgb.Booster):
        feature_names = [
            "BPHIGH4","TOLDHI2","CVDINFR4","CVDSTRK3","_BMI5",
            "GENHLTH","SMOKE100","EXERANY2","ALCDAY5","SEX",
            "_AGE_G","_RACE","EDUCA","INCOME2","HLTHPLN1","year",
        ]
        dmat = xgb.DMatrix(X, feature_names=feature_names)
        return (model.predict(dmat) >= 0.5).astype(int)
    return model.predict(X)


print("Loading dataset ...")
t0 = time.time()
X, y = load_data(DATA_PATH)
load_time = time.time() - t0
print(f"  {len(y):,} rows loaded in {load_time:.1f}s\n")

results = {}
lines   = []

header = f"{'Model':<22}  {'Accuracy':>8}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}  {'Time(s)':>8}"
sep    = "-" * len(header)
print(header)
print(sep)

for name, pkl_file in MODELS.items():
    pkl_path = os.path.join(MODELS_DIR, pkl_file)
    print(f"  Loading {name} ...", flush=True)
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)

    t_start = time.time()
    if name == "XGBoost":
        preds = predict_xgb(model, X)
    else:
        preds = model.predict(X)
    elapsed = time.time() - t_start

    metrics = evaluate(preds, y)
    results[name] = {"metrics": metrics, "elapsed": elapsed}

    row = (f"  {name:<20}  {metrics['accuracy']:8.4f}  "
           f"{metrics['precision']:9.4f}  {metrics['recall']:6.4f}  "
           f"{metrics['f1']:6.4f}  {elapsed:8.1f}")
    print(row)
    lines.append(row)

print(sep)
print()

with open(OUTPUT_FILE, "w") as f:
    f.write("Serial baseline results\n")
    f.write(f"Dataset: {DATA_PATH}  ({len(y):,} rows)\n\n")
    f.write(header + "\n" + sep + "\n")
    for line in lines:
        f.write(line + "\n")
    f.write(sep + "\n")

print(f"Results written to {OUTPUT_FILE}")
print("Run the Hadoop jobs and compare their wall-clock times against the")
print("'Time(s)' column above to quantify the parallel speedup.")
