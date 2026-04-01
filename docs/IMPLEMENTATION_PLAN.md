# Customer Churn Project - Implementation Plan

## Objective
Build a production-style machine learning project for telecom customer churn prediction by strictly following `Customer_Churn_Project_Instructions.md` and using only the IBM Telco dataset:
- Expected file name: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Expected location: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

No synthetic dataset will be used at any step.

## Execution Order

### Phase 1 - Project Scaffolding
1. Create the folder structure.
2. Create dependency and config files.
3. Create starter README with setup instructions.

### Phase 2 - Data Ingestion and Validation
1. Implement data loading utility from `data/raw`.
2. Validate required columns and dataset shape expectations.
3. Fail fast with a clear message if dataset is missing.

### Phase 3 - EDA
1. Dataset overview (`shape`, `info`, null counts, class distribution).
2. Required plots:
   - Churn distribution
   - Churn by contract type
   - Churn by tenure bins
   - Churn by monthly charges bins
   - Correlation heatmap for numeric features
3. Save all artifacts in `reports/figures` and summaries in `reports`.

### Phase 4 - Preprocessing
1. Drop `customerID`.
2. Fix `TotalCharges` (`to_numeric`, coercion, median imputation).
3. Encode target (`Churn`: Yes/No -> 1/0).
4. One-hot encode categorical features.
5. Build train/test split using `stratify=y`.
6. Build preprocessing transformers for numeric and categorical features.

### Phase 5 - Modeling
1. Train baseline models:
   - Logistic Regression (`class_weight='balanced'`)
   - Random Forest (`class_weight='balanced'`)
2. Train boosted model:
   - XGBoost (or fallback to LightGBM if unavailable)
3. Hyperparameter tuning using `RandomizedSearchCV` with `scoring='f1'` and `cv=5`.

### Phase 6 - Evaluation
1. Evaluate with:
   - Accuracy
   - Precision (churn class)
   - Recall (churn class)
   - F1-score (churn class)
   - ROC-AUC
   - Confusion matrix
2. Plot and save:
   - Confusion matrix for best model
   - ROC curves for all models
   - Feature importance (top 10)
3. Save metrics to `reports/metrics_summary.csv`.

### Phase 7 - Deliverables
1. Save best model (`artifacts/models/best_model.pkl`).
2. Save preprocessing bundle and metadata (`artifacts/models/`).
3. Create notebook with full workflow.
4. Finalize README with exact run commands and interpretation.

## Strict Dataset Policy
- Only use Kaggle IBM Telco dataset from instruction file.
- If file is not present locally and cannot be downloaded automatically in this environment, user will be asked to manually download and place it at `data/raw/`.

## Completion Criteria
- Project is runnable end-to-end.
- Required artifacts are generated in expected folders.
- No synthetic data usage.
- Workflow and outputs align with instruction targets.
