from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "bank_transactions_data_2.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FINAL_RESULT_PATH = OUTPUT_DIR / "bank_transaction_anomaly_final_result.csv"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance_xgboost_surrogate.csv"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
PREPROCESSING_ARTIFACTS_PATH = ARTIFACT_DIR / "preprocessing_artifacts.pkl"
IF_MODEL_PATH = ARTIFACT_DIR / "isolation_forest_model.pkl"
LOF_MODEL_PATH = ARTIFACT_DIR / "lof_novelty_model.pkl"
OCSVM_MODEL_PATH = ARTIFACT_DIR / "one_class_svm_model.pkl"
MCD_MODEL_PATH = ARTIFACT_DIR / "mcd_model.pkl"
NUMERIC_SCALER_PATH = ARTIFACT_DIR / "numeric_scaler.pkl"
RISK_SCORE_SCALER_PATH = ARTIFACT_DIR / "risk_score_scaler.pkl"

REQUIRED_PREDICTION_ARTIFACTS = [
    PREPROCESSING_ARTIFACTS_PATH,
    IF_MODEL_PATH,
    LOF_MODEL_PATH,
    OCSVM_MODEL_PATH,
    MCD_MODEL_PATH,
    NUMERIC_SCALER_PATH,
    RISK_SCORE_SCALER_PATH,
]

RISK_ORDER = ["Low", "Low-Medium", "Medium", "High"]

RISK_COLOR_MAP = {
    "Low": "#2F7D5B",
    "Low-Medium": "#B7791F",
    "Medium": "#C86432",
    "High": "#C93C4D",
}

CHART_COLOR_SEQUENCE = [
    "#2E5D8A",
    "#5F7F9F",
    "#7E9BB8",
    "#B7791F",
    "#C93C4D",
    "#7A8491",
]

DEFAULT_BAR_COLOR = "#2E5D8A"
