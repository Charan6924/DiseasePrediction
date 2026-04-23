import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib
from tqdm import tqdm
import os

MODELS_DIR = "/home/cxv166/DiseasePrediction/models"
os.makedirs(MODELS_DIR, exist_ok=True)

CSV_PATH = "/home/cxv166/DiseasePrediction/brfss_diabetes_clean.csv"   
TARGET_COL = "DIABETE3"           
TEST_SIZE = 0.20
RANDOM_STATE = 42

df = pd.read_csv(CSV_PATH)
print(df.head())

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=11),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "XGBoost":XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
}

for name, model in tqdm(models.items(), desc="Training models", unit="model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*50}")
    print(f"{name}")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    joblib.dump(model, f"{MODELS_DIR}/{name.replace(' ', '_')}.pkl")
    print(f"Saved: {MODELS_DIR}/{name.replace(' ', '_')}.pkl")
