# Project Final Summary

## Project
Customer Churn Prediction using IBM Telco dataset (Kaggle).

## Dataset
- Source: blastchar/telco-customer-churn (Kaggle)
- File used: data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
- Synthetic data usage: None

## Best Models By Purpose
- Accuracy: catboost_tuned
- Recall: random_forest_tuned
- F1: random_forest_tuned
- ROC-AUC: catboost_tuned

## Key Performance (Latest Run)
- random_forest_tuned: Accuracy 0.7637 | Precision 0.5375 | Recall 0.7861 | F1 0.6384 | AUC 0.8428
- lightgbm_tuned: Accuracy 0.7594 | Precision 0.5321 | Recall 0.7754 | F1 0.6311 | AUC 0.8399
- catboost_tuned: Accuracy 0.8062 | Precision 0.6784 | Recall 0.5134 | F1 0.5845 | AUC 0.8476

## Business Conclusion
- For retention campaigns, prioritize random_forest_tuned due to higher churn recall and F1.
- For maximum overall accuracy and ranking quality, use catboost_tuned.

## Implemented Features
- End-to-end training pipeline with EDA, preprocessing, tuning, and evaluation
- Threshold tuning and selected threshold export
- SMOTE vs class-weight comparison
- Multi-model training (LogReg, RF, XGBoost, LightGBM, CatBoost, Ensemble)
- Streamlit app with:
  - objective-based selection
  - manual model selection
  - threshold strategy modes
  - model performance table

## Run Commands
- Pipeline: $env:PYTHONPATH='src'; c:/python314/python.exe run_pipeline.py
- App: c:/python314/python.exe -m streamlit run streamlit_app.py
