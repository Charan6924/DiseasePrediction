
"""

PySpark inference benchmark — runs a pickled sklearn/xgboost model in parallel

across Spark executors and reports accuracy, precision, recall, F1, and timing.



Usage:

    spark-submit --files models/<MODEL>.pkl spark_inference.py <model_name>



Where <model_name> is one of:

    Logistic_Regression, Random_Forest, XGBoost, KNN

"""

import sys

import time

import pickle

import numpy as np

from pyspark.sql import SparkSession



DATA_PATH = "data/brfss_diabetes_clean.csv"

MODELS_DIR = "models"



MODEL_FILES = {

    "Logistic_Regression": "Logistic_Regression.pkl",

    "Random_Forest":       "Random_Forest.pkl",

    "XGBoost":             "XGBoost.pkl",

    "KNN":                 "KNN.pkl",

}



FEATURE_INDICES = list(range(1, 17))

FEATURE_NAMES = [

    "BPHIGH4", "TOLDHI2", "CVDINFR4", "CVDSTRK3", "_BMI5",

    "GENHLTH", "SMOKE100", "EXERANY2", "ALCDAY5", "SEX",

    "_AGE_G", "_RACE", "EDUCA", "INCOME2", "HLTHPLN1", "year",

]



if len(sys.argv) != 2:

    print("Usage: spark-submit spark_inference.py <model_name>")

    print(f"Available: {list(MODEL_FILES.keys())}")

    sys.exit(1)



model_name = sys.argv[1]

if model_name not in MODEL_FILES:

    print(f"Unknown model '{model_name}'. Choose from: {list(MODEL_FILES.keys())}")

    sys.exit(1)



pkl_filename = MODEL_FILES[model_name]

print(f"=== PySpark inference: {model_name} ===")

print(f"Model file: {pkl_filename}")

print(f"Data file:  {DATA_PATH}")





def predict_partition(rows):

    import pickle

    import numpy as np

    from pyspark import SparkFiles



    # Spark distributes the .pkl via --files; SparkFiles.get() returns the

    # correct local path on each executor.

    model_path_local = SparkFiles.get(pkl_filename)

    with open(model_path_local, "rb") as f:

        model = pickle.load(f)



    is_xgb_booster = False

    if model_name == "XGBoost":

        import xgboost as xgb

        is_xgb_booster = isinstance(model, xgb.Booster)



    BATCH = 512

    feats_buf, actuals_buf = [], []



    def flush():

        if not feats_buf:

            return

        X = np.array(feats_buf, dtype=np.float64)

        if is_xgb_booster:

            import xgboost as xgb

            dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES)

            preds = (model.predict(dmat) >= 0.5).astype(int)

        else:

            preds = model.predict(X)

        out = list(zip(preds, actuals_buf))

        feats_buf.clear()

        actuals_buf.clear()

        return out



    for line in rows:

        line = line.strip()

        if not line or line.startswith("DIABETE3"):

            continue

        parts = line.split(",")

        if len(parts) < 17:

            continue

        try:

            actual = int(float(parts[0]))

            features = [float(parts[i]) for i in FEATURE_INDICES]

        except (ValueError, IndexError):

            continue

        actuals_buf.append(actual)

        feats_buf.append(features)

        if len(feats_buf) >= BATCH:

            for p, a in flush():

                yield int(p), int(a)



    final = flush()

    if final:

        for p, a in final:

            yield int(p), int(a)





spark = (

    SparkSession.builder

    .appName(f"DiabetesInference-{model_name}")

    .config("spark.driver.memory", "4g")

    .config("spark.executor.memory", "4g")

    .getOrCreate()

)

sc = spark.sparkContext



print(f"\nSpark version: {spark.version}")

print(f"Default parallelism: {sc.defaultParallelism}")



t0 = time.time()

lines_rdd = sc.textFile(DATA_PATH, minPartitions=16)

print(f"Input partitions: {lines_rdd.getNumPartitions()}")



preds_rdd = lines_rdd.mapPartitions(predict_partition)





def confusion_counts(pred_actual):

    p, a = pred_actual

    if   p == 1 and a == 1: return (1, 0, 0, 0)

    elif p == 0 and a == 0: return (0, 1, 0, 0)

    elif p == 1 and a == 0: return (0, 0, 1, 0)

    else:                   return (0, 0, 0, 1)





def add_tuples(x, y):

    return tuple(a + b for a, b in zip(x, y))





tp, tn, fp, fn = preds_rdd.map(confusion_counts).reduce(add_tuples)

elapsed = time.time() - t0



total = tp + tn + fp + fn

acc  = (tp + tn) / total if total else 0

prec = tp / (tp + fp)    if (tp + fp) else 0

rec  = tp / (tp + fn)    if (tp + fn) else 0

f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0



print("\n=== Evaluation Results ===")

print(f"Model         : {model_name}")

print(f"Total samples : {total}")

print(f"Accuracy      : {acc:.4f}")

print(f"Precision     : {prec:.4f}")

print(f"Recall        : {rec:.4f}")

print(f"F1 Score      : {f1:.4f}")

print()

print("Confusion Matrix:")

print(f"               Pred=0   Pred=1")

print(f"  Actual=0     {tn:6d}   {fp:6d}")

print(f"  Actual=1     {fn:6d}   {tp:6d}")

print()

print(f"Wall-clock time: {elapsed:.2f}s")



spark.stop()

