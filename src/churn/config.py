from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT_DIR / "artifacts" / "models"

TARGET_COL = "Churn"
ID_COL = "customerID"
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
