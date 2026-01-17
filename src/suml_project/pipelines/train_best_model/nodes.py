from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay


def train_final_model(
    best_model: Any,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
) -> Any:
    X_full = pd.concat([X_train, X_val], ignore_index=True)
    y_full = pd.concat([y_train, y_val], ignore_index=True).values.ravel()

    best_model.fit(X_full, y_full)

    return best_model


def evaluate_on_test(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    y_true = y_test.values.ravel()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
    }

    predictions = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
    })

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Predicted_0", "Predicted_1"])

    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Rain", "Rain"])
    ax.set_yticklabels(["No Rain", "Rain"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix - Test Set")
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    return metrics, predictions, cm_df
