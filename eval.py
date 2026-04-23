import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve,f1_score, precision_score, recall_score, accuracy_score,ConfusionMatrixDisplay)

def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Model": name,
        "ROC-AUC": (roc_auc_score(y_test, y_prob), 4),
        "Recall": (recall_score(y_test, y_pred), 4),
        "F1": (f1_score(y_test, y_pred), 4),
        "Precision": (precision_score(y_test, y_pred), 4),
        "Accuracy": (accuracy_score(y_test, y_pred), 4),
        "_prob": y_prob,
        "_pred": y_pred,
    }


def print_summary(results):
    df = pd.DataFrame(results).drop(columns=["_prob", "_pred"])
    df = df.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    print("  Model Comparison")
    print(df.to_string(index=False))


def plot_roc_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["_prob"])
        ax.plot(fpr, tpr, label=f"{r['Model']} (AUC={r['ROC-AUC']})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curves")
    ax.legend()
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150)
    plt.show()
    print("Saved: roc_curves.png")


def plot_confusion_matrices(results, y_test):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        ConfusionMatrixDisplay.from_predictions(
            y_test, r["_pred"],
            display_labels=["No Diabetes", "Diabetes"],
            ax=ax, colorbar=False
        )
        ax.set_title(r["Model"], fontsize=11)
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150)
    plt.show()
    print("Saved: confusion_matrices.png")


