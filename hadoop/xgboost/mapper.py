#!/usr/bin/env python3
"""
Mapper for XGBoost.

XGBoost.pkl may be either an XGBClassifier (sklearn wrapper) or a native
xgb.Booster.  Both are handled below: the sklearn wrapper's .predict()
returns class labels directly; the native Booster's .predict() returns
probabilities that we threshold at 0.5.

CSV column layout (0-indexed):
  0        : DIABETE3  (target label, 0 or 1)
  1 - 16   : features  BPHIGH4, TOLDHI2, CVDINFR4, CVDSTRK3, _BMI5,
                       GENHLTH, SMOKE100, EXERANY2, ALCDAY5, SEX,
                       _AGE_G, _RACE, EDUCA, INCOME2, HLTHPLN1, year
"""
import sys
import pickle
import numpy as np

MODEL_FILE = "XGBoost.pkl"
FEATURE_INDICES = list(range(1, 17))

FEATURE_NAMES = [
    "BPHIGH4", "TOLDHI2", "CVDINFR4", "CVDSTRK3", "_BMI5",
    "GENHLTH", "SMOKE100", "EXERANY2", "ALCDAY5", "SEX",
    "_AGE_G", "_RACE", "EDUCA", "INCOME2", "HLTHPLN1", "year",
]

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# Detect native Booster vs sklearn wrapper once at startup
import xgboost as xgb
_is_booster = isinstance(model, xgb.Booster)

def predict(features_2d):
    if _is_booster:
        dmat = xgb.DMatrix(features_2d, feature_names=FEATURE_NAMES)
        prob = model.predict(dmat)
        return (prob >= 0.5).astype(int)
    return model.predict(features_2d)

for line in sys.stdin:
    line = line.strip()
    if not line or line.startswith("DIABETE3"):
        continue
    parts = line.split(",")
    if len(parts) < 17:
        continue
    try:
        actual = int(float(parts[0]))
        features = np.array([float(parts[i]) for i in FEATURE_INDICES],
                            dtype=np.float64).reshape(1, -1)
        pred = int(predict(features)[0])
        print(f"{pred}\t{actual}")
    except (ValueError, IndexError):
        continue
