from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import ID_COL, NUMERIC_COLS, TARGET_COL


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer


def prepare_data(df: pd.DataFrame) -> PreparedData:
    work = df.copy()
    work["TotalCharges"] = pd.to_numeric(work["TotalCharges"], errors="coerce")
    work[TARGET_COL] = work[TARGET_COL].map({"Yes": 1, "No": 0})

    if ID_COL in work.columns:
        work = work.drop(columns=[ID_COL])

    X = work.drop(columns=[TARGET_COL])
    y = work[TARGET_COL]

    numeric_cols = [c for c in NUMERIC_COLS if c in X.columns]
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
    )
