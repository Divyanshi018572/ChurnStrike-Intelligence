from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy import sparse
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline


def run_smote_vs_class_weight(preprocessor, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    rows = []

    class_weight_model = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=42)),
        ]
    )
    class_weight_model.fit(X_train, y_train)
    cw_pred = class_weight_model.predict(X_test)
    cw_prob = class_weight_model.predict_proba(X_test)[:, 1]
    rows.append(
        {
            "strategy": "class_weight_balanced",
            "accuracy": accuracy_score(y_test, cw_pred),
            "precision_churn": precision_score(y_test, cw_pred, zero_division=0),
            "recall_churn": recall_score(y_test, cw_pred, zero_division=0),
            "f1_churn": f1_score(y_test, cw_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, cw_prob),
        }
    )

    smote_preprocessor = clone(preprocessor)
    X_train_t = smote_preprocessor.fit_transform(X_train)
    X_test_t = smote_preprocessor.transform(X_test)

    if sparse.issparse(X_train_t):
        X_train_t = X_train_t.toarray()
    if sparse.issparse(X_test_t):
        X_test_t = X_test_t.toarray()

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_t, y_train)
    smote_model = LogisticRegression(max_iter=2500, random_state=42)
    smote_model.fit(X_res, y_res)

    sm_pred = smote_model.predict(X_test_t)
    sm_prob = smote_model.predict_proba(X_test_t)[:, 1]
    rows.append(
        {
            "strategy": "smote_logistic",
            "accuracy": accuracy_score(y_test, sm_pred),
            "precision_churn": precision_score(y_test, sm_pred, zero_division=0),
            "recall_churn": recall_score(y_test, sm_pred, zero_division=0),
            "f1_churn": f1_score(y_test, sm_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, sm_prob),
        }
    )

    return pd.DataFrame(rows).sort_values(["f1_churn", "recall_churn"], ascending=False)


def threshold_sweep(model, X_test, y_test, threshold_min=0.3, threshold_max=0.7, step=0.02):
    if not hasattr(model, "predict_proba"):
        return pd.DataFrame(), None

    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(threshold_min, threshold_max + 1e-9, step)
    rows = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        rows.append(
            {
                "threshold": float(round(t, 3)),
                "accuracy": accuracy_score(y_test, pred),
                "precision_churn": precision_score(y_test, pred, zero_division=0),
                "recall_churn": recall_score(y_test, pred, zero_division=0),
                "f1_churn": f1_score(y_test, pred, zero_division=0),
            }
        )

    results = pd.DataFrame(rows)
    feasible = results[results["precision_churn"] >= 0.5]
    if not feasible.empty:
        best = feasible.sort_values(["recall_churn", "f1_churn"], ascending=False).iloc[0]
    else:
        best = results.sort_values(["f1_churn", "recall_churn"], ascending=False).iloc[0]

    return results.sort_values("threshold"), best.to_dict()
