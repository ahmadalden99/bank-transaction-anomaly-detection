import pandas as pd

from src.config import FEATURE_IMPORTANCE_PATH, FINAL_RESULT_PATH, RISK_ORDER


def load_transaction_results(path=FINAL_RESULT_PATH):
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    dataframe = pd.read_csv(path)

    if "RiskLevel" in dataframe.columns:
        dataframe["RiskLevel"] = pd.Categorical(
            dataframe["RiskLevel"],
            categories=RISK_ORDER,
            ordered=True,
        )

    return dataframe


def load_feature_importance(path=FEATURE_IMPORTANCE_PATH):
    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()
