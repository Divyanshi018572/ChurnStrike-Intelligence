from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from churn.config import FIGURES_DIR, REPORTS_DIR


def run_eda(df: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "rows": len(df),
        "columns": df.shape[1],
        "churn_rate_pct": round((df["Churn"].eq("Yes").mean() * 100), 2),
    }
    pd.DataFrame([summary]).to_csv(REPORTS_DIR / "eda_summary.csv", index=False)
    df.isnull().sum().to_csv(REPORTS_DIR / "missing_values.csv", header=["missing_count"])

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Churn")
    plt.title("Churn Distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "churn_distribution.png", dpi=200)
    plt.close()

    contract_rate = df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean())
    plt.figure(figsize=(8, 5))
    sns.barplot(x=contract_rate.index, y=contract_rate.values)
    plt.title("Churn Rate by Contract")
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "churn_by_contract.png", dpi=200)
    plt.close()

    tmp = df.copy()
    tmp["TotalCharges"] = pd.to_numeric(tmp["TotalCharges"], errors="coerce")
    tmp["tenure_bin"] = pd.cut(tmp["tenure"], bins=[0, 12, 24, 48, 72], include_lowest=True)
    tenure_rate = tmp.groupby("tenure_bin", observed=False)["Churn"].apply(lambda x: (x == "Yes").mean())

    plt.figure(figsize=(8, 5))
    sns.barplot(x=tenure_rate.index.astype(str), y=tenure_rate.values)
    plt.title("Churn Rate by Tenure Bin")
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "churn_by_tenure_bin.png", dpi=200)
    plt.close()

    tmp["monthly_bin"] = pd.qcut(tmp["MonthlyCharges"], q=4, duplicates="drop")
    monthly_rate = tmp.groupby("monthly_bin", observed=False)["Churn"].apply(lambda x: (x == "Yes").mean())

    plt.figure(figsize=(9, 5))
    sns.barplot(x=monthly_rate.index.astype(str), y=monthly_rate.values)
    plt.title("Churn Rate by Monthly Charges Quartile")
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "churn_by_monthly_charges.png", dpi=200)
    plt.close()

    corr_df = tmp[["tenure", "MonthlyCharges", "TotalCharges"]].copy()
    corr_df = corr_df.fillna(corr_df.median(numeric_only=True))
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_df.corr(), annot=True, cmap="Blues", fmt=".2f")
    plt.title("Correlation Heatmap (Numeric)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()
