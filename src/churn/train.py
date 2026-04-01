from __future__ import annotations

from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


def _get_xgboost_pipeline(preprocessor):
    try:
        from xgboost import XGBClassifier

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=42,
                        n_estimators=250,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    except Exception:
        return None


def _get_lightgbm_pipeline(preprocessor):
    try:
        from lightgbm import LGBMClassifier

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LGBMClassifier(
                        objective="binary",
                        n_estimators=300,
                        learning_rate=0.05,
                        num_leaves=31,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
    except Exception:
        return None


def _get_catboost_pipeline(preprocessor):
    try:
        from catboost import CatBoostClassifier

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    CatBoostClassifier(
                        iterations=300,
                        depth=6,
                        learning_rate=0.05,
                        loss_function="Logloss",
                        verbose=False,
                        allow_writing_files=False,
                        random_seed=42,
                    ),
                ),
            ]
        )
    except Exception:
        return None


def build_models(preprocessor) -> Dict[str, Pipeline]:
    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=12,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    xgb = _get_xgboost_pipeline(preprocessor)
    if xgb is not None:
        models["xgboost"] = xgb
    lgbm = _get_lightgbm_pipeline(preprocessor)
    if lgbm is not None:
        models["lightgbm"] = lgbm
    catboost = _get_catboost_pipeline(preprocessor)
    if catboost is not None:
        models["catboost"] = catboost
    return models


def fit_models(models, X_train, y_train):
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


def tune_random_forest(rf_pipeline, X_train, y_train):
    param_dist = {
        "model__n_estimators": [100, 150, 200, 250, 300],
        "model__max_depth": [5, 8, 10, 12, 15, 20],
        "model__min_samples_split": [2, 4, 6, 8, 10],
        "model__min_samples_leaf": [1, 2, 3, 4],
    }
    search = RandomizedSearchCV(
        rf_pipeline,
        param_distributions=param_dist,
        n_iter=20,
        scoring="f1",
        cv=5,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_xgboost(xgb_pipeline, X_train, y_train):
    param_dist = {
        "model__n_estimators": [100, 200, 300, 400, 500],
        "model__max_depth": [3, 4, 5, 6, 7],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }
    search = RandomizedSearchCV(
        xgb_pipeline,
        param_distributions=param_dist,
        n_iter=20,
        scoring="f1",
        cv=5,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_lightgbm(lgbm_pipeline, X_train, y_train):
    param_dist = {
        "model__n_estimators": [150, 250, 350, 500],
        "model__num_leaves": [15, 31, 63],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "model__min_child_samples": [10, 20, 30, 50],
    }
    search = RandomizedSearchCV(
        lgbm_pipeline,
        param_distributions=param_dist,
        n_iter=20,
        scoring="f1",
        cv=5,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_catboost(cat_pipeline, X_train, y_train):
    param_dist = {
        "model__iterations": [200, 300, 500],
        "model__depth": [4, 5, 6, 7, 8],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__l2_leaf_reg": [1, 3, 5, 7, 9],
    }
    search = RandomizedSearchCV(
        cat_pipeline,
        param_distributions=param_dist,
        n_iter=16,
        scoring="f1",
        cv=5,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def build_soft_voting_ensemble(fitted_models):
    estimators = []
    for name in ["random_forest_tuned", "xgboost_tuned", "lightgbm_tuned", "catboost_tuned"]:
        if name in fitted_models:
            estimators.append((name, fitted_models[name]))

    if len(estimators) < 2:
        return None

    # Equal weights work well as a stable first ensemble baseline.
    return VotingClassifier(estimators=estimators, voting="soft", weights=[1.0] * len(estimators), n_jobs=-1)
