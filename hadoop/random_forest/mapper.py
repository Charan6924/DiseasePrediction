#!/usr/bin/env python3
"""
Mapper for Random Forest.

The Random_Forest.pkl is ~1.9 GB; Hadoop distributes it to each task node
via -files so every mapper loads its own local copy.  Because the file is
large, the job uses fewer mappers than the LR/XGB jobs — set
-D mapreduce.job.maps=<N> in run.sh to tune memory vs parallelism.

CSV column layout (0-indexed):
  0        : DIABETE3  (target label, 0 or 1)
  1 - 16   : features  BPHIGH4, TOLDHI2, CVDINFR4, CVDSTRK3, _BMI5,
                       GENHLTH, SMOKE100, EXERANY2, ALCDAY5, SEX,
                       _AGE_G, _RACE, EDUCA, INCOME2, HLTHPLN1, year
"""
import sys
import pickle
import numpy as np

MODEL_FILE = "Random_Forest.pkl"
FEATURE_INDICES = list(range(1, 17))

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
