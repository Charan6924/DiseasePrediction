#!/usr/bin/env python3
"""
Hadoop Streaming mapper: load trained model, emit predictions per row.

Input  (stdin):  CSV lines from cleaned_data.csv
Output (stdout): "actual\tpredicted,probability" per line

Column layout (from Owen's preprocessing):
  [0]     DIABETE3  <- target label
  [1-15]            <- clinical features (BPHIGH4 ... HLTHPLN1)
  [16]    year      <- included as feature (models were trained with it, n_features_in_=16)
"""
import sys
import joblib

HEADER_PREFIX = "DIABETE3"
FEATURE_COLS = list(range(1, 17))  # cols 1..16 inclusive — 16 features matching n_features_in_

def load_model():
    # model.pkl is distributed to each worker via -files flag in the streaming job
    return joblib.load("model.pkl")

def main():
    model = load_model()

    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith(HEADER_PREFIX):
            continue

        fields = line.split(",")
        if len(fields) < 17:
            continue

        try:
            actual = int(float(fields[0]))
            features = [float(fields[i]) for i in FEATURE_COLS]
        except ValueError:
            continue

        pred = int(model.predict([features])[0])

        # predict_proba is not available on all sklearn estimators
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba([features])[0][1]
        else:
            prob = float(pred)

        # Key = actual label so all outputs route to a single reducer
        # Value = predicted,probability
        print(f"{actual}\t{pred},{prob:.4f}")

if __name__ == "__main__":
    main()
