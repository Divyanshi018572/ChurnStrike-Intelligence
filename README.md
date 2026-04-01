# Customer Churn Prediction (IBM Telco)

This project predicts whether a telecom customer will churn (`Churn = Yes/No`) using the real IBM Telco dataset from Kaggle.

Dataset used (strict):
- `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Source: `blastchar/telco-customer-churn` on Kaggle

No synthetic dataset is used.

## Project Structure

- `docs/Customer_Churn_Project_Instructions.md`: given instruction file
- `docs/IMPLEMENTATION_PLAN.md`: execution plan
- `data/raw/`: original dataset
- `src/churn/`: modular ML pipeline code
- `run_pipeline.py`: one-command training/evaluation run
- `reports/`: tables and evaluation outputs
- `reports/figures/`: EDA and model plots
- `artifacts/models/`: saved model (`best_model.pkl`)
- `notebooks/`: step-by-step notebook

Deployment-focused additions:
- `.streamlit/config.toml`: Streamlit runtime configuration
- `Procfile`: web process entrypoint for PaaS
- `Dockerfile`: container build recipe
- `.dockerignore`: Docker build context cleanup
- `scripts/start_app.ps1`: Windows app start script
- `scripts/start_app.sh`: Linux/Mac app start script
- `docs/DEPLOYMENT.md`: deployment checklist and steps

## Setup

1. Install dependencies

```bash
c:/python314/python.exe -m pip install -r requirements.txt
```

2. Confirm dataset exists

- Required file: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Run Full Pipeline

From project root:

```bash
$env:PYTHONPATH = "src"
c:/python314/python.exe run_pipeline.py
```

This will generate:
- `reports/eda_summary.csv`
- `reports/missing_values.csv`
- `reports/metrics_summary.csv`
- `reports/threshold_tuning.csv`
- `reports/selected_threshold.json`
- `reports/smote_vs_class_weight.csv`
- `reports/classification_report_best_model.json`
- `reports/tuning_results.json` (if tuning runs)
- `reports/feature_importance_top10.csv` (if tree best model)
- `reports/figures/*.png`
- `artifacts/models/best_model.pkl`

## Step-by-Step What the Pipeline Does

1. Data Loading
- Reads the Telco CSV from `data/raw`.
- Fails immediately if file is missing.

2. EDA
- Checks churn distribution.
- Plots churn by contract type.
- Plots churn by tenure bins.
- Plots churn by monthly charge quartiles.
- Builds numeric correlation heatmap.

3. Preprocessing
- Converts `TotalCharges` to numeric.
- Drops `customerID`.
- Encodes target `Churn` as `Yes->1`, `No->0`.
- Uses train/test split with `stratify=y`.
- Uses `ColumnTransformer`:
  - Numeric: median imputation + scaling
  - Categorical: most-frequent imputation + one-hot encoding

4. Modeling
- Baseline: Logistic Regression (`class_weight='balanced'`)
- Strong baseline: Random Forest (`class_weight='balanced'`)
- Final candidate: XGBoost (if installed and available)

5. Hyperparameter Tuning
- RandomizedSearchCV (`cv=5`, `scoring='f1'`) for Random Forest.
- RandomizedSearchCV (`cv=5`, `scoring='f1'`) for XGBoost if available.

6. Evaluation
- Accuracy, Precision, Recall, F1, ROC-AUC.
- ROC curve comparison for all trained models.
- Confusion matrix for best model.
- Feature importance top-10 (tree model only).

7. Threshold Tuning
- Sweeps thresholds from 0.30 to 0.70.
- Chooses a recommended threshold with recall-first logic while keeping precision practical.
- Saves full sweep in `reports/threshold_tuning.csv` and selected threshold in `reports/selected_threshold.json`.

8. SMOTE vs Class-Weight Comparison
- Runs a focused logistic regression comparison:
  - Class-weight strategy (`class_weight='balanced'`)
  - SMOTE resampling strategy
- Saves comparison in `reports/smote_vs_class_weight.csv`.

## Run Streamlit App

```bash
c:/python314/python.exe -m streamlit run streamlit_app.py
```

The app loads:
- Trained model from `artifacts/models/best_model.pkl`
- Suggested threshold from `reports/selected_threshold.json` (if available)
- Feature defaults from `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Notes

- Recall is prioritized for churn class (missing churners is costly).
- If `xgboost` is unavailable in the environment, pipeline still runs using Logistic Regression and Random Forest.

## Final Results (Latest Run)

Top models from `reports/metrics_summary.csv`:

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| random_forest_tuned | 0.7637 | 0.5375 | 0.7861 | 0.6384 | 0.8428 |
| lightgbm_tuned | 0.7594 | 0.5321 | 0.7754 | 0.6311 | 0.8399 |
| catboost_tuned | 0.8062 | 0.6784 | 0.5134 | 0.5845 | 0.8476 |

Purpose-based best model recommendations (`reports/model_recommendations.json`):
- Best for Accuracy: `catboost_tuned`
- Best for Recall: `random_forest_tuned`
- Best for F1: `random_forest_tuned`
- Best for AUC: `catboost_tuned`

Recommended deployment choice:
- Retention-focused operations: `random_forest_tuned` (higher recall/F1 on churn)
- Accuracy-focused scoring: `catboost_tuned`

## Submission Checklist

- [x] Real Kaggle dataset used (no synthetic data)
- [x] End-to-end pipeline (`run_pipeline.py`) runs successfully
- [x] EDA outputs generated in `reports/figures/`
- [x] Hyperparameter tuning performed
- [x] Threshold tuning and SMOTE comparison added
- [x] Multi-model training and comparison report generated
- [x] Purpose-based model recommendations generated
- [x] Trained models exported to `artifacts/models/`
- [x] Streamlit app with model selection and threshold strategies implemented
- [x] Notebook workflow included in `notebooks/customer_churn_workflow.ipynb`

## How To Demo In 3 Minutes

1. Show project objective (20-30 sec)
- Explain: "This project predicts telecom customer churn so retention teams can act before customers leave."
- Mention strict real-data usage from Kaggle IBM Telco dataset.

2. Run pipeline and show outputs (45-60 sec)
- Command: `$env:PYTHONPATH='src'; c:/python314/python.exe run_pipeline.py`
- Show generated artifacts in `reports/` and `artifacts/models/`.
- Highlight `metrics_summary.csv` and `model_recommendations.json`.

3. Explain best model by purpose (40-50 sec)
- Accuracy/AUC best: `catboost_tuned`
- Recall/F1 best: `random_forest_tuned`
- Business reasoning: recall-first is often preferred for churn retention.

4. Open Streamlit app and do one live prediction (50-60 sec)
- Command: `c:/python314/python.exe -m streamlit run streamlit_app.py`
- In app, show:
  - Auto-by-objective model selection
  - Manual model selection
  - Threshold strategy modes
- Enter one sample customer and click **Predict Churn**.

5. Close with business impact (15-20 sec)
- "The app supports strategy-based deployment: recall-focused retention mode or accuracy-focused scoring mode."
