
"""

Serial baseline — runs models one row at a time using pure Python (no Spark).

This is the true "1x" reference for the paper's parallel speedup analysis.



Usage:

    python3 serial_baseline.py [model_name]



If no model name given, runs all four. Prints accuracy + wall-clock time.

Times the prediction phase only — model loading and CSV parsing are

excluded from the timing because Spark also amortizes those costs.

"""

import sys

import time

import pickle

import os

import numpy as np



DATA_PATH  = "data/brfss_diabetes_clean.csv"

MODELS_DIR = "models"

RESULTS_FILE = "serial_timing_results.txt"



MODEL_FILES = {

    "Logistic_Regression": "Logistic_Regression.pkl",

    "Random_Forest":       "Random_Forest.pkl",

    "XGBoost":             "XGBoost.pkl",

    "KNN":                 "K-Nearest_Neighbors.pkl",

}



FEATURE_INDICES = list(range(1, 17))

FEATURE_NAMES = [

    "BPHIGH4", "TOLDHI2", "CVDINFR4", "CVDSTRK3", "_BMI5",

    "GENHLTH", "SMOKE100", "EXERANY2", "ALCDAY5", "SEX",

    "_AGE_G", "_RACE", "EDUCA", "INCOME2", "HLTHPLN1", "year",

]





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

    total = tp + tn + fp + fn

    acc  = (tp + tn) / total if total else 0

    prec = tp / (tp + fp)    if (tp + fp) else 0

    rec  = tp / (tp + fn)    if (tp + fn) else 0

    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0

    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, total=total,

                tp=tp, tn=tn, fp=fp, fn=fn)





def predict_xgb(model, X):

    import xgboost as xgb

    if isinstance(model, xgb.Booster):

        dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES)

        return (model.predict(dmat) >= 0.5).astype(int)

    return model.predict(X)





# Determine which models to run

if len(sys.argv) >= 2:

    requested = [sys.argv[1]]

    if requested[0] not in MODEL_FILES:

        print(f"Unknown model '{requested[0]}'. Choose from: {list(MODEL_FILES.keys())}")

        sys.exit(1)

else:

    requested = list(MODEL_FILES.keys())



# Load data once

print("Loading dataset ...")

t_load_start = time.time()

X, y = load_data(DATA_PATH)

load_time = time.time() - t_load_start

print(f"  {len(y):,} rows loaded in {load_time:.1f}s\n")



results = {}



header = f"{'Model':<22}  {'Accuracy':>8}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}  {'Time(s)':>8}"

sep = "-" * len(header)

print(header)

print(sep)



for name in requested:

    pkl_file = MODEL_FILES[name]

    pkl_path = os.path.join(MODELS_DIR, pkl_file)



    try:

        with open(pkl_path, "rb") as f:

            model = pickle.load(f)

    except Exception as e:

        print(f"  {name:<20}  -- LOAD ERROR: {e}")

        continue



    # Time only the prediction phase

    t_start = time.time()

    try:

        if name == "XGBoost":

            preds = predict_xgb(model, X)

        else:

            preds = model.predict(X)

    except Exception as e:

        print(f"  {name:<20}  -- PREDICT ERROR: {e}")

        continue

    elapsed = time.time() - t_start



    metrics = evaluate(preds, y)

    results[name] = {"metrics": metrics, "elapsed": elapsed}



    row = (f"  {name:<20}  {metrics['accuracy']:8.4f}  "

           f"{metrics['precision']:9.4f}  {metrics['recall']:6.4f}  "

           f"{metrics['f1']:6.4f}  {elapsed:8.2f}")

    print(row)



print(sep)

print()



# Append to a results file for easy comparison later

with open(RESULTS_FILE, "a") as f:

    f.write(f"\n=== Serial baseline run ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===\n")

    f.write(f"Dataset: {DATA_PATH}  ({len(y):,} rows)\n")

    f.write(header + "\n" + sep + "\n")

    for name, info in results.items():

        m = info["metrics"]

        f.write(f"  {name:<20}  {m['accuracy']:8.4f}  {m['precision']:9.4f}  "

                f"{m['recall']:6.4f}  {m['f1']:6.4f}  {info['elapsed']:8.2f}\n")

    f.write(sep + "\n")



print(f"Results appended to {RESULTS_FILE}")

