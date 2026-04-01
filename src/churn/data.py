from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn.config import DATA_PATH


def load_data(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    return pd.read_csv(csv_path)
