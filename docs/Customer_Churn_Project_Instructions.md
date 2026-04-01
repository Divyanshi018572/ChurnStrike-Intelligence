**PROJECT INSTRUCTION FILE**

**Customer Churn Prediction**

Machine Learning Project  |  Classification  |  Telecom Domain

|<p>Dataset</p><p>**Telco IBM (Kaggle)**</p>|<p>Rows</p><p>**7,043 records**</p>|<p>Difficulty</p><p>**Medium**</p>|<p>Target Accuracy</p><p>**79 - 85%**</p>|
| :-: | :-: | :-: | :-: |


# **1. Project Overview**
Customer churn prediction is one of the most impactful and widely deployed machine learning applications in the telecom, SaaS, banking, and subscription industries. The objective is to predict whether a customer will stop using a service (churn = Yes) before it actually happens, allowing the business to take proactive retention steps.

|**Real-World Use Case**|
| :- |
|Telecom companies lose 15-25% of their customers annually due to churn. Retaining an existing customer is 5x cheaper than acquiring a new one. Companies like Airtel, Vodafone, and Jio actively use churn prediction models to trigger retention campaigns, offer discounts, or assign dedicated account managers to at-risk customers.|

# **2. Dataset Details**
**Source**

- Name: IBM Telco Customer Churn Dataset
- Platform: Kaggle — https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Format: CSV — single file, no joins required
- License: Public / Open Use

**Dataset Statistics**

|**Property**|**Value**|
| :- | :- |
|Total Rows|7,043 customers|
|Total Columns|21 features|
|Target Column|Churn (Yes / No)|
|Class Distribution|~73% No Churn, ~27% Churn (imbalanced)|
|Missing Values|11 rows in TotalCharges (fill or drop)|
|Data Types|Mix of categorical, binary, and numeric|

**Key Features**

- customerID — unique identifier (drop before modeling)
- gender, SeniorCitizen, Partner, Dependents — demographics
- tenure — months customer has been with company (very important)
- PhoneService, MultipleLines, InternetService — service subscriptions
- OnlineSecurity, TechSupport, StreamingTV, StreamingMovies — add-ons
- Contract — Month-to-month, One year, Two year (top predictor)
- PaperlessBilling, PaymentMethod — billing info
- MonthlyCharges, TotalCharges — financial features
- Churn — TARGET: Yes or No

# **3. Step-by-Step Workflow**
## **Step 1 — Environment Setup**
Install the required Python libraries before starting:

|pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn matplotlib seaborn|
| :- |

## **Step 2 — Load & Explore Data (EDA)**
1. Load CSV with pandas: df = pd.read\_csv('WA\_Fn-UseC\_-Telco-Customer-Churn.csv')
1. Check shape, dtypes, null values: df.info(), df.isnull().sum()
1. Convert TotalCharges to numeric (it loads as string due to spaces)
1. Plot churn distribution — confirm ~27% churn rate
1. Plot churn rate by Contract type, tenure, MonthlyCharges
1. Correlation heatmap for numeric features

|**Key EDA Finding**|
| :- |
|Month-to-month contract customers churn at ~43% rate vs 11% for one-year and 3% for two-year contracts. Tenure < 12 months = highest churn risk. These two features alone give strong signal.|

## **Step 3 — Data Preprocessing**
1. Drop customerID column (not a feature)
1. Encode binary columns (Yes/No) using LabelEncoder or map({'Yes':1,'No':0})
1. One-hot encode multi-class categoricals: Contract, PaymentMethod, InternetService
1. Fix TotalCharges: pd.to\_numeric(df['TotalCharges'], errors='coerce'), then fillna with median
1. Scale numeric features (tenure, MonthlyCharges, TotalCharges) using StandardScaler for SVM/KNN; tree models do not require scaling
1. Split data: X\_train, X\_test, y\_train, y\_test = train\_test\_split(X, y, test\_size=0.2, random\_state=42, stratify=y)

## **Step 4 — Handle Class Imbalance**
The dataset has ~73% No Churn vs ~27% Churn. This is a mild imbalance but must be handled:

- Option A — SMOTE (Synthetic Minority Oversampling): from imblearn.over\_sampling import SMOTE — generates synthetic churn samples
- Option B — class\_weight='balanced' in sklearn models — automatically adjusts weights
- Option C — Adjust classification threshold from 0.5 to 0.4 to catch more churners
- Recommended: Use class\_weight='balanced' for baseline, then try SMOTE for boosted models

## **Step 5 — Model Building**

|**Model**|**When to Use**|**Expected Accuracy**|
| :- | :- | :- |
|Logistic Regression|Baseline model, interpretable, fast|78 - 80%|
|Decision Tree|Visual explanation, easy to debug|76 - 79%|
|Random Forest|Strong baseline, handles noise well|80 - 82%|
|XGBoost|Best performer for tabular data|82 - 85%|
|LightGBM|Faster than XGBoost, similar accuracy|81 - 84%|
|SVM (RBF kernel)|Good for small-medium datasets|79 - 82%|

Recommended training order: Start with Logistic Regression as baseline → Random Forest → XGBoost as final model.

## **Step 6 — Hyperparameter Tuning**
1. Use GridSearchCV or RandomizedSearchCV with cv=5 (5-fold cross validation)
1. XGBoost key params to tune: n\_estimators (100-500), max\_depth (3-7), learning\_rate (0.01-0.3), subsample (0.7-1.0)
1. Random Forest key params: n\_estimators (100-300), max\_depth (5-20), min\_samples\_split (2-10)
1. Use scoring='f1' or scoring='roc\_auc' — NOT 'accuracy' — when tuning for imbalanced data

## **Step 7 — Evaluate the Model**
Never rely on accuracy alone for imbalanced classification. Use all of these:

|**Metric**|**What it Measures**|**Target Value**|
| :- | :- | :- |
|Accuracy|Overall correct predictions|79 - 85%|
|Precision (Churn)|Of predicted churners, how many actually churned|> 70%|
|Recall (Churn)|Of actual churners, how many did we catch|> 75%|
|F1-Score (Churn)|Harmonic mean of Precision and Recall|> 72%|
|AUC-ROC|Model's ability to separate classes|> 0.83|
|Confusion Matrix|Full breakdown of TP, TN, FP, FN|Visualize always|

|**Important: Recall > Precision for this problem**|
| :- |
|In churn prediction, missing a churner (False Negative) is more costly than falsely flagging a loyal customer (False Positive). A loyal customer gets an unnecessary discount; a churner who is missed leaves permanently. Always optimize for Recall on the churn class.|

# **4. Feature Importance**
After training XGBoost or Random Forest, extract and plot feature importances:

|**Rank**|**Feature**|**Importance Level**|**Business Insight**|
| :- | :- | :- | :- |
|1|Contract Type|Very High|Month-to-month customers churn 4x more|
|2|tenure|Very High|Customers in first 6 months are highest risk|
|3|MonthlyCharges|High|High bill = higher churn probability|
|4|TotalCharges|High|Correlated with tenure, use carefully|
|5|InternetService (Fiber)|Medium-High|Fiber users churn more than DSL|
|6|OnlineSecurity (No)|Medium|Customers without security add-on churn more|
|7|TechSupport (No)|Medium|No support = lower satisfaction|
|8|PaperlessBilling|Medium|Slightly higher churn rate|
|9|PaymentMethod|Low-Medium|Electronic check users churn more|
|10|SeniorCitizen|Low|Minor signal, limited in dataset|

# **5. Expected Results Summary**

|**Metric**|**Baseline (Logistic Reg.)**|**Best Model (XGBoost)**|
| :- | :- | :- |
|Accuracy|78 - 80%|82 - 85%|
|Precision (Churn)|63 - 68%|71 - 76%|
|Recall (Churn)|69 - 73%|76 - 80%|
|F1-Score (Churn)|66 - 70%|73 - 78%|
|AUC-ROC|0\.80 - 0.83|0\.84 - 0.87|

# **6. Common Mistakes to Avoid**
- Using accuracy as the only metric — with 73/27 split, a model predicting all 'No Churn' gets 73% accuracy but is useless
- Not applying stratify=y in train\_test\_split — can cause skewed class distribution in train/test
- Forgetting to drop customerID — it is not a feature and will cause data leakage if kept
- Scaling data and then applying SMOTE — always apply SMOTE AFTER splitting, BEFORE or AFTER scaling (use pipeline)
- Overfitting with deep decision trees — use max\_depth <= 10 and prune using min\_samples\_leaf
- Not checking for multicollinearity — TotalCharges = tenure x MonthlyCharges, consider dropping TotalCharges

# **7. Recommended Tools & Libraries**

|**Library**|**Purpose**|
| :- | :- |
|pandas|Data loading, cleaning, manipulation|
|numpy|Numerical operations|
|scikit-learn|Preprocessing, models, metrics, cross-validation|
|xgboost|Gradient boosted trees — best classifier|
|lightgbm|Fast gradient boosting alternative|
|imbalanced-learn|SMOTE and other resampling techniques|
|matplotlib / seaborn|EDA plots, confusion matrix, ROC curve|
|shap|Model explainability — show why a customer was flagged|
|joblib|Save and load trained model to disk|

# **8. Project Deliverables Checklist**
- Jupyter Notebook (.ipynb) with full EDA, preprocessing, modeling, and evaluation
- Trained model saved as .pkl file using joblib
- Classification Report + Confusion Matrix visualization
- ROC Curve plot comparing all models
- Feature Importance bar chart (top 10 features)
- README.md with project description, setup steps, and results summary
- Optional: Streamlit app for live churn prediction on new customer input


Customer Churn Prediction  |  ML Project Instruction File  |  Classification Project #1
