from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from churn.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR


def evaluate(models, X_test, y_test):
    rows = []
    roc_curves = {}

    for name, model in models.items():
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else pred

        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, pred),
                "precision_churn": precision_score(y_test, pred, zero_division=0),
                "recall_churn": recall_score(y_test, pred, zero_division=0),
                "f1_churn": f1_score(y_test, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, prob),
            }
        )
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_curves[name] = (fpr, tpr)

    metrics_df = pd.DataFrame(rows).sort_values(["f1_churn", "roc_auc"], ascending=False)
    return metrics_df, roc_curves


def _build_model_recommendations(metrics_df: pd.DataFrame) -> dict:
    by_accuracy = metrics_df.sort_values("accuracy", ascending=False).iloc[0]["model"]
    by_recall = metrics_df.sort_values("recall_churn", ascending=False).iloc[0]["model"]
    by_f1 = metrics_df.sort_values("f1_churn", ascending=False).iloc[0]["model"]
    by_auc = metrics_df.sort_values("roc_auc", ascending=False).iloc[0]["model"]
    return {
        "best_for_accuracy": str(by_accuracy),
        "best_for_recall": str(by_recall),
        "best_for_f1": str(by_f1),
        "best_for_auc": str(by_auc),
    }


def save_outputs(best_name, best_model, metrics_df, roc_curves, X_test, y_test, tuning_log, fitted_models):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(REPORTS_DIR / "metrics_summary.csv", index=False)
    recommendations = _build_model_recommendations(metrics_df)
    with open(REPORTS_DIR / "model_recommendations.json", "w", encoding="utf-8") as f:
        json.dump(recommendations, f, indent=2)

    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr) in roc_curves.items():
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curves.png", dpi=200)
    plt.close()

    best_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_best_model.png", dpi=200)
    plt.close()

    cls_report = classification_report(y_test, best_pred, output_dict=True, zero_division=0)
    with open(REPORTS_DIR / "classification_report_best_model.json", "w", encoding="utf-8") as f:
        json.dump(cls_report, f, indent=2)

    if tuning_log:
        with open(REPORTS_DIR / "tuning_results.json", "w", encoding="utf-8") as f:
            json.dump(tuning_log, f, indent=2)

    for name, model in fitted_models.items():
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")

    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")


def save_feature_importance(best_model):
    model = best_model.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return

    preprocessor = best_model.named_steps["preprocessor"]
    names = preprocessor.get_feature_names_out()
    imp = pd.DataFrame({"feature": names, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    top10 = imp.head(10)
    top10.to_csv(REPORTS_DIR / "feature_importance_top10.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top10, x="importance", y="feature")
    plt.title("Top 10 Feature Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance_top10.png", dpi=200)
    plt.close()
