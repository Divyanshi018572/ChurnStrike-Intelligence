from __future__ import annotations

import json

from churn.data import load_data
from churn.eda import run_eda
from churn.evaluate import evaluate, save_feature_importance, save_outputs
from churn.experiments import run_smote_vs_class_weight, threshold_sweep
from churn.preprocess import prepare_data
from churn.train import (
    build_models,
    build_soft_voting_ensemble,
    fit_models,
    tune_catboost,
    tune_lightgbm,
    tune_random_forest,
    tune_xgboost,
)


def run_pipeline() -> None:
    df = load_data()
    run_eda(df)

    prep = prepare_data(df)
    models = build_models(prep.preprocessor)
    fitted = fit_models(models, prep.X_train, prep.y_train)

    tuning_log = {}
    if "random_forest" in fitted:
        tuned_rf, rf_params, rf_score = tune_random_forest(fitted["random_forest"], prep.X_train, prep.y_train)
        fitted["random_forest_tuned"] = tuned_rf
        tuning_log["random_forest_tuned"] = {"best_params": rf_params, "best_cv_f1": rf_score}

    if "xgboost" in fitted:
        tuned_xgb, xgb_params, xgb_score = tune_xgboost(fitted["xgboost"], prep.X_train, prep.y_train)
        fitted["xgboost_tuned"] = tuned_xgb
        tuning_log["xgboost_tuned"] = {"best_params": xgb_params, "best_cv_f1": xgb_score}

    if "lightgbm" in fitted:
        tuned_lgbm, lgbm_params, lgbm_score = tune_lightgbm(fitted["lightgbm"], prep.X_train, prep.y_train)
        fitted["lightgbm_tuned"] = tuned_lgbm
        tuning_log["lightgbm_tuned"] = {"best_params": lgbm_params, "best_cv_f1": lgbm_score}

    if "catboost" in fitted:
        tuned_cat, cat_params, cat_score = tune_catboost(fitted["catboost"], prep.X_train, prep.y_train)
        fitted["catboost_tuned"] = tuned_cat
        tuning_log["catboost_tuned"] = {"best_params": cat_params, "best_cv_f1": cat_score}

    ensemble = build_soft_voting_ensemble(fitted)
    if ensemble is not None:
        ensemble.fit(prep.X_train, prep.y_train)
        fitted["soft_voting_ensemble"] = ensemble

    metrics_df, roc_curves = evaluate(fitted, prep.X_test, prep.y_test)
    best_name = str(metrics_df.iloc[0]["model"])
    best_model = fitted[best_name]

    smote_cmp_df = run_smote_vs_class_weight(
        prep.preprocessor,
        prep.X_train,
        prep.y_train,
        prep.X_test,
        prep.y_test,
    )
    smote_cmp_df.to_csv("reports/smote_vs_class_weight.csv", index=False)

    threshold_df, best_threshold = threshold_sweep(best_model, prep.X_test, prep.y_test)
    if not threshold_df.empty:
        threshold_df.to_csv("reports/threshold_tuning.csv", index=False)
    if best_threshold is not None:
        with open("reports/selected_threshold.json", "w", encoding="utf-8") as f:
            json.dump(best_threshold, f, indent=2)

    save_outputs(
        best_name,
        best_model,
        metrics_df,
        roc_curves,
        prep.X_test,
        prep.y_test,
        tuning_log,
        fitted,
    )
    save_feature_importance(best_model)


if __name__ == "__main__":
    run_pipeline()
