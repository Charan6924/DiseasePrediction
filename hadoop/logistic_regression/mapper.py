#!/usr/bin/env python3
"""
Mapper for Logistic Regression.

Input  : CSV rows read from stdin (Hadoop splits the file across mappers).
Output : tab-separated  pred<TAB>actual  for every row.

The model pkl is distributed to each task node via the Hadoop -files flag and
lands in the mapper's current working directory.

CSV column layout (0-indexed):
  0        : DIABETE3  (target label, 0 or 1)
  1 - 16   : features  BPHIGH4, TOLDHI2, CVDINFR4, CVDSTRK3, _BMI5,
                       GENHLTH, SMOKE100, EXERANY2, ALCDAY5, SEX,
                       _AGE_G, _RACE, EDUCA, INCOME2, HLTHPLN1, year
"""
import sys
import pickle
import numpy as np

MODEL_FILE = "Logistic_Regression.pkl"
FEATURE_INDICES = list(range(1, 17))   # columns 1-16 inclusive

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

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
        pred = int(model.predict(features)[0])
        print(f"{pred}\t{actual}")
    except (ValueError, IndexError):
        continue
