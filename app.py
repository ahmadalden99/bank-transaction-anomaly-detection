import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path


st.set_page_config(
    page_title="Bank Transaction Anomaly Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


DATA_PATH = Path("outputs/bank_transaction_anomaly_final_result.csv")
FEATURE_IMPORTANCE_PATH = Path("outputs/feature_importance_xgboost_surrogate.csv")

ARTIFACT_DIR = Path("artifacts")
PREPROCESSING_ARTIFACTS_PATH = ARTIFACT_DIR / "preprocessing_artifacts.pkl"
IF_MODEL_PATH = ARTIFACT_DIR / "isolation_forest_model.pkl"
LOF_MODEL_PATH = ARTIFACT_DIR / "lof_novelty_model.pkl"
OCSVM_MODEL_PATH = ARTIFACT_DIR / "one_class_svm_model.pkl"
MCD_MODEL_PATH = ARTIFACT_DIR / "mcd_model.pkl"
NUMERIC_SCALER_PATH = ARTIFACT_DIR / "numeric_scaler.pkl"
RISK_SCORE_SCALER_PATH = ARTIFACT_DIR / "risk_score_scaler.pkl"


RISK_ORDER = ["Low", "Low-Medium", "Medium", "High"]

RISK_COLOR_MAP = {
    "Low": "#34C759",
    "Low-Medium": "#F4B740",
    "Medium": "#F47B3E",
    "High": "#E5484D",
}

CHART_COLOR_SEQUENCE = ["#6C63E8", "#5BC7E8", "#34C759", "#F4B740", "#E5484D", "#8E93A8"]
DEFAULT_BAR_COLOR = "#6C63E8"


st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: #F7F8FC;
        color: #111827;
    }

    [data-testid="stHeader"] {
        background-color: rgba(247, 248, 252, 0.96);
    }

    .block-container {
        max-width: 1260px;
        padding-top: 1.15rem;
        padding-bottom: 2rem;
        padding-left: 1.75rem;
        padding-right: 1.75rem;
    }

    [data-testid="stSidebar"] {
        background-color: #5F56D9;
        border-right: 1px solid #5149C7;
        width: 272px !important;
        min-width: 272px !important;
    }

    [data-testid="stSidebar"] h1 {
        color: #FFFFFF;
        font-size: 17px;
        font-weight: 760;
        letter-spacing: 0;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #E7E4FF;
    }

    [data-testid="stSidebar"] label {
        color: #FFFFFF;
        font-size: 12px;
        font-weight: 640;
    }

    [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.14);
        border-color: rgba(255, 255, 255, 0.26);
        border-radius: 8px;
        box-shadow: none;
    }

    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
        background-color: #FFFFFF;
        border: 1px solid rgba(255, 255, 255, 0.52);
        border-radius: 6px;
        margin-top: 2px;
        margin-bottom: 2px;
    }

    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] span {
        color: #4F46C8;
        font-size: 12px;
        font-weight: 640;
    }

    [data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] {
        color: #E7E4FF;
    }

    .page-header {
        background-color: #FFFFFF;
        border: 1px solid #E7EAF4;
        border-left: 4px solid #6C63E8;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 16px;
        box-shadow: 0 8px 22px rgba(31, 41, 55, 0.035);
    }

    .header-eyebrow {
        color: #6C63E8;
        font-size: 11px;
        font-weight: 760;
        letter-spacing: 0;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .main-title {
        color: #111827;
        font-size: 26px;
        font-weight: 760;
        line-height: 1.18;
        margin-bottom: 7px;
        letter-spacing: 0;
    }

    .main-subtitle {
        color: #4B5563;
        font-size: 13px;
        max-width: 880px;
        line-height: 1.55;
    }

    .header-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 10px 22px;
        margin-top: 12px;
        color: #4B5563;
        font-size: 12px;
    }

    .header-meta strong {
        color: #111827;
        font-weight: 760;
    }

    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E7EAF4;
        padding: 12px 14px;
        border-radius: 10px;
        box-shadow: 0 6px 18px rgba(31, 41, 55, 0.035);
        min-height: 86px;
    }

    .metric-label {
        font-size: 12px;
        color: #667085;
        font-weight: 660;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 23px;
        color: #111827;
        font-weight: 760;
        letter-spacing: 0;
        line-height: 1.2;
    }

    .metric-note {
        font-size: 11px;
        color: #7C8494;
        margin-top: 5px;
    }

    .section-title {
        font-size: 18px;
        font-weight: 740;
        color: #111827;
        margin-top: 8px;
        margin-bottom: 8px;
        letter-spacing: 0;
    }

    .section-subtitle {
        font-size: 13px;
        color: #667085;
        margin-bottom: 12px;
    }

    .subsection-title {
        color: #111827;
        font-size: 13px;
        font-weight: 700;
        margin: 4px 0 12px 0;
        letter-spacing: 0;
    }

    .info-box {
        background-color: #FFFFFF;
        border: 1px solid #E7EAF4;
        padding: 12px 14px;
        border-radius: 10px;
        color: #344054;
        font-size: 13px;
        line-height: 1.55;
    }

    .risk-low {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #BBF7D0;
        padding: 5px 9px;
        border-radius: 6px;
        font-weight: 750;
        display: inline-block;
    }

    .risk-lowmedium {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
        padding: 5px 9px;
        border-radius: 6px;
        font-weight: 750;
        display: inline-block;
    }

    .risk-medium {
        background-color: #FFEDD5;
        color: #9A3412;
        border: 1px solid #FED7AA;
        padding: 5px 9px;
        border-radius: 6px;
        font-weight: 750;
        display: inline-block;
    }

    .risk-high {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FECACA;
        padding: 5px 9px;
        border-radius: 6px;
        font-weight: 750;
        display: inline-block;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #E7EAF4;
        border-radius: 10px;
        overflow: hidden;
    }

    div[data-testid="stPlotlyChart"] {
        background-color: #FFFFFF;
        border: 1px solid #E7EAF4;
        border-radius: 10px;
        padding: 6px;
        box-shadow: 0 6px 18px rgba(31, 41, 55, 0.03);
    }

    div[data-testid="stTabs"] button {
        color: #4B5563;
        font-size: 13px;
        font-weight: 620;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #5F56D9;
    }

    div[data-baseweb="tab-highlight"] {
        background-color: #5F56D9;
    }

    div[data-testid="stButton"] button,
    div[data-testid="stDownloadButton"] button {
        border-radius: 6px;
        border: 1px solid #5F56D9;
        background-color: #5F56D9;
        color: #FFFFFF;
        font-weight: 640;
    }

    div[data-testid="stButton"] button:hover,
    div[data-testid="stDownloadButton"] button:hover {
        border-color: #4F46C8;
        background-color: #4F46C8;
        color: #FFFFFF;
    }

    div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        background-color: #5F56D9;
        border-color: #5F56D9;
        box-shadow: none;
    }

    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
        background-color: #5F56D9;
    }

    input,
    textarea,
    div[data-baseweb="select"] > div {
        border-radius: 8px;
        box-shadow: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(f"File tidak ditemukan: {DATA_PATH}")
        st.stop()

    df_loaded = pd.read_csv(DATA_PATH)

    if "RiskLevel" in df_loaded.columns:
        df_loaded["RiskLevel"] = pd.Categorical(
            df_loaded["RiskLevel"],
            categories=RISK_ORDER,
            ordered=True,
        )

    return df_loaded


@st.cache_data
def load_feature_importance():
    if FEATURE_IMPORTANCE_PATH.exists():
        return pd.read_csv(FEATURE_IMPORTANCE_PATH)
    return pd.DataFrame()


@st.cache_resource
def load_prediction_artifacts():
    required_paths = [
        PREPROCESSING_ARTIFACTS_PATH,
        IF_MODEL_PATH,
        LOF_MODEL_PATH,
        OCSVM_MODEL_PATH,
        MCD_MODEL_PATH,
        NUMERIC_SCALER_PATH,
        RISK_SCORE_SCALER_PATH,
    ]

    missing_paths = [str(path) for path in required_paths if not path.exists()]

    if len(missing_paths) > 0:
        return None, missing_paths

    artifacts = joblib.load(PREPROCESSING_ARTIFACTS_PATH)

    models = {
        "if_model": joblib.load(IF_MODEL_PATH),
        "lof_model": joblib.load(LOF_MODEL_PATH),
        "ocsvm_model": joblib.load(OCSVM_MODEL_PATH),
        "mcd_model": joblib.load(MCD_MODEL_PATH),
        "numeric_scaler": joblib.load(NUMERIC_SCALER_PATH),
        "risk_score_scaler": joblib.load(RISK_SCORE_SCALER_PATH),
    }

    return {
        "artifacts": artifacts,
        "models": models,
    }, []


def format_number(value):
    return f"{value:,.0f}"


def format_percentage(value):
    return f"{value:.2f}%"


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge_html(risk_level):
    risk_text = str(risk_level)

    if risk_text == "High":
        css_class = "risk-high"
    elif risk_text == "Medium":
        css_class = "risk-medium"
    elif risk_text == "Low-Medium":
        css_class = "risk-lowmedium"
    else:
        css_class = "risk-low"

    return f'<span class="{css_class}">{risk_text}</span>'


def safe_columns(dataframe, columns):
    return [col for col in columns if col in dataframe.columns]


def style_plotly_chart(fig, height=380, showlegend=True):
    fig.update_layout(
        template="plotly_white",
        colorway=CHART_COLOR_SEQUENCE,
        height=height,
        margin=dict(l=12, r=12, t=44, b=12),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#344054", size=12),
        title=dict(font=dict(color="#111827", size=13), x=0, xanchor="left"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0,
            title=None,
            font=dict(size=11),
        ),
        showlegend=showlegend,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#F0F3FA",
        zeroline=False,
        linecolor="#E7EAF4",
        tickfont=dict(size=11, color="#4B5563"),
        title_font=dict(size=12, color="#4B5563"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#F0F3FA",
        zeroline=False,
        linecolor="#E7EAF4",
        tickfont=dict(size=11, color="#4B5563"),
        title_font=dict(size=12, color="#4B5563"),
    )
    fig.update_traces(marker_line_width=0, opacity=0.88, selector=dict(type="bar"))
    return fig


def stringify_value_column(table, value_column="Value"):
    table = table.copy()
    if value_column in table.columns:
        table[value_column] = table[value_column].apply(
            lambda value: "" if pd.isna(value) else str(value)
        )
    return table


def assign_balance_group(transaction_type, account_balance, balance_thresholds):
    if transaction_type in balance_thresholds.index:
        low_boundary = balance_thresholds.loc[transaction_type, "Low_to_Medium_Boundary"]
        high_boundary = balance_thresholds.loc[transaction_type, "Medium_to_High_Boundary"]

        if account_balance <= low_boundary:
            return "Low"
        elif account_balance <= high_boundary:
            return "Medium"
        else:
            return "High"

    return "Medium"


def get_group_average_amount(transaction_type, channel, balance_group, avg_amount_by_group):
    matched_group = avg_amount_by_group[
        (avg_amount_by_group["TransactionType"].astype(str) == str(transaction_type))
        & (avg_amount_by_group["Channel"].astype(str) == str(channel))
        & (avg_amount_by_group["BalanceGroup_ByType"].astype(str) == str(balance_group))
    ]

    if len(matched_group) > 0:
        return float(matched_group["Avg_TransactionAmount"].iloc[0])

    fallback_type_channel = avg_amount_by_group[
        (avg_amount_by_group["TransactionType"].astype(str) == str(transaction_type))
        & (avg_amount_by_group["Channel"].astype(str) == str(channel))
    ]

    if len(fallback_type_channel) > 0:
        return float(fallback_type_channel["Avg_TransactionAmount"].mean())

    return float(avg_amount_by_group["Avg_TransactionAmount"].mean())


def assign_risk_level_from_score(score):
    if score >= 75:
        return "High"
    elif score >= 50:
        return "Medium"
    elif score >= 25:
        return "Low-Medium"
    else:
        return "Low"


def build_new_transaction_features(input_data, prediction_bundle):
    artifacts = prediction_bundle["artifacts"]
    models = prediction_bundle["models"]

    numeric_features = artifacts["numeric_features"]
    categorical_features = artifacts["categorical_features"]
    model_columns = artifacts["model_columns"]
    duration_threshold = artifacts["duration_threshold"]
    login_threshold = artifacts["login_threshold"]
    balance_thresholds = artifacts["balance_thresholds"]
    frequency_maps = artifacts["frequency_maps"]
    avg_amount_by_group = artifacts["avg_amount_by_group"]

    transaction_date = pd.to_datetime(input_data["TransactionDate"])

    hour = transaction_date.hour
    day_of_week = transaction_date.dayofweek
    month = transaction_date.month
    is_weekend = 1 if day_of_week in [5, 6] else 0

    transaction_amount = float(input_data["TransactionAmount"])
    account_balance = float(input_data["AccountBalance"])

    amount_to_balance_ratio = transaction_amount / account_balance if account_balance != 0 else 0

    balance_group = assign_balance_group(
        input_data["TransactionType"],
        account_balance,
        balance_thresholds,
    )

    avg_amount = get_group_average_amount(
        input_data["TransactionType"],
        input_data["Channel"],
        balance_group,
        avg_amount_by_group,
    )

    amount_to_group_avg_ratio = transaction_amount / avg_amount if avg_amount != 0 else 0

    account_tx_count = frequency_maps.get("AccountID", {}).get(input_data["AccountID"], 1)
    device_tx_count = frequency_maps.get("DeviceID", {}).get(input_data["DeviceID"], 1)
    merchant_tx_count = frequency_maps.get("MerchantID", {}).get(input_data["MerchantID"], 1)
    ip_tx_count = frequency_maps.get("IP Address", {}).get(input_data["IP Address"], 1)

    raw_row = {
        "CustomerAge": int(input_data["CustomerAge"]),
        "TransactionAmount": transaction_amount,
        "AccountBalance": account_balance,
        "TransactionDuration": int(input_data["TransactionDuration"]),
        "LoginAttempts": int(input_data["LoginAttempts"]),
        "Hour_sin": np.sin(2 * np.pi * hour / 24),
        "Hour_cos": np.cos(2 * np.pi * hour / 24),
        "DayOfWeek_sin": np.sin(2 * np.pi * day_of_week / 7),
        "DayOfWeek_cos": np.cos(2 * np.pi * day_of_week / 7),
        "Month_sin": np.sin(2 * np.pi * month / 12),
        "Month_cos": np.cos(2 * np.pi * month / 12),
        "IsWeekend": is_weekend,
        "Amount_to_Balance_Ratio": amount_to_balance_ratio,
        "Amount_to_TypeChannelBalanceGroupAvg_Ratio": amount_to_group_avg_ratio,
        "IsHighLoginAttempts": 1 if int(input_data["LoginAttempts"]) >= login_threshold else 0,
        "IsLongDuration": 1 if int(input_data["TransactionDuration"]) > duration_threshold else 0,
        "AccountTxCount": account_tx_count,
        "DeviceTxCount": device_tx_count,
        "MerchantTxCount": merchant_tx_count,
        "IPTxCount": ip_tx_count,
        "TransactionType": input_data["TransactionType"],
        "Channel": input_data["Channel"],
        "Location": input_data["Location"],
        "CustomerOccupation": input_data["CustomerOccupation"],
        "BalanceGroup_ByType": balance_group,
    }

    df_new_raw = pd.DataFrame([raw_row])

    df_new_numeric_raw = df_new_raw[numeric_features].copy()
    df_new_categorical_raw = df_new_raw[categorical_features].copy()

    df_new_numeric_scaled = pd.DataFrame(
        models["numeric_scaler"].transform(df_new_numeric_raw),
        columns=numeric_features,
    )

    df_new_categorical_encoded = pd.get_dummies(
        df_new_categorical_raw,
        columns=categorical_features,
        drop_first=False,
        dtype=int,
    )

    X_new = pd.concat(
        [
            df_new_numeric_scaled,
            df_new_categorical_encoded,
        ],
        axis=1,
    )

    X_new = X_new.reindex(columns=model_columns, fill_value=0)
    X_new = X_new.astype(float)

    return df_new_raw, X_new


def predict_new_transaction(input_data, prediction_bundle):
    models = prediction_bundle["models"]

    df_new_raw, X_new = build_new_transaction_features(input_data, prediction_bundle)

    if_pred = models["if_model"].predict(X_new)[0]
    lof_pred = models["lof_model"].predict(X_new)[0]
    ocsvm_pred = models["ocsvm_model"].predict(X_new)[0]
    mcd_pred = models["mcd_model"].predict(X_new)[0]

    if_score = -models["if_model"].decision_function(X_new)[0]
    lof_score = -models["lof_model"].decision_function(X_new)[0]
    ocsvm_score = -models["ocsvm_model"].decision_function(X_new)[0]
    mcd_score = -models["mcd_model"].decision_function(X_new)[0]

    anomaly_flags = {
        "IF_Anomaly": 1 if if_pred == -1 else 0,
        "LOF_Anomaly": 1 if lof_pred == -1 else 0,
        "OCSVM_Anomaly": 1 if ocsvm_pred == -1 else 0,
        "MCD_Anomaly": 1 if mcd_pred == -1 else 0,
    }

    anomaly_vote_count = sum(anomaly_flags.values())
    ensemble_anomaly = 1 if anomaly_vote_count >= 2 else 0
    vote_score = anomaly_vote_count / 4

    score_df = pd.DataFrame(
        [[if_score, lof_score, ocsvm_score, mcd_score]],
        columns=["IF_Score", "LOF_Score", "OCSVM_Score", "MCD_Score"],
    )

    normalized_scores = models["risk_score_scaler"].transform(score_df)
    normalized_scores = np.clip(normalized_scores, 0, 1)

    average_anomaly_score_norm = float(normalized_scores.mean())

    risk_score = (
        (0.7 * vote_score)
        + (0.3 * average_anomaly_score_norm)
    ) * 100

    risk_score = round(float(risk_score), 2)
    risk_level = assign_risk_level_from_score(risk_score)

    prediction_result = {
        "IF_Anomaly": anomaly_flags["IF_Anomaly"],
        "LOF_Anomaly": anomaly_flags["LOF_Anomaly"],
        "OCSVM_Anomaly": anomaly_flags["OCSVM_Anomaly"],
        "MCD_Anomaly": anomaly_flags["MCD_Anomaly"],
        "IF_Score": if_score,
        "LOF_Score": lof_score,
        "OCSVM_Score": ocsvm_score,
        "MCD_Score": mcd_score,
        "AnomalyVoteCount": anomaly_vote_count,
        "Ensemble_Anomaly": ensemble_anomaly,
        "VoteScore": vote_score,
        "Average_AnomalyScore_Norm": average_anomaly_score_norm,
        "RiskScore": risk_score,
        "RiskLevel": risk_level,
    }

    return df_new_raw, prediction_result


df = load_data()
feature_importance = load_feature_importance()

total_dataset_transactions = len(df)
total_dataset_anomaly = int(df["Ensemble_Anomaly"].sum()) if "Ensemble_Anomaly" in df.columns else 0
total_dataset_high_risk = (
    int((df["RiskLevel"].astype(str) == "High").sum())
    if "RiskLevel" in df.columns
    else 0
)

st.markdown(
    f"""
    <div class="page-header">
        <div class="header-eyebrow">Fraud analytics workspace</div>
        <div class="main-title">Bank Transaction Risk Monitor</div>
        <div class="main-subtitle">
            Pantau transaksi tidak biasa, skor risiko, voting model, dan faktor utama
            dari pipeline unsupervised anomaly detection.
        </div>
        <div class="header-meta">
            <span><strong>{total_dataset_transactions:,}</strong> transactions</span>
            <span><strong>{total_dataset_anomaly:,}</strong> ensemble anomalies</span>
            <span><strong>{total_dataset_high_risk:,}</strong> high-risk cases</span>
            <span>4-model unsupervised ensemble</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.sidebar.title("Filters")
st.sidebar.caption("Refine transactions by risk, channel, customer profile, and score range.")

risk_options = [
    risk for risk in RISK_ORDER
    if risk in df["RiskLevel"].astype(str).unique().tolist()
]

selected_risk = st.sidebar.multiselect(
    "Risk Level",
    options=risk_options,
    default=risk_options,
)

transaction_type_options = sorted(df["TransactionType"].dropna().astype(str).unique().tolist())
selected_transaction_type = st.sidebar.multiselect(
    "Transaction Type",
    options=transaction_type_options,
    default=transaction_type_options,
)

channel_options = sorted(df["Channel"].dropna().astype(str).unique().tolist())
selected_channel = st.sidebar.multiselect(
    "Channel",
    options=channel_options,
    default=channel_options,
)

occupation_options = sorted(df["CustomerOccupation"].dropna().astype(str).unique().tolist())
selected_occupation = st.sidebar.multiselect(
    "Customer Occupation",
    options=occupation_options,
    default=occupation_options,
)

location_options = sorted(df["Location"].dropna().astype(str).unique().tolist())
selected_location = st.sidebar.multiselect(
    "Location",
    options=location_options,
    default=location_options,
)

min_score = float(df["RiskScore"].min())
max_score = float(df["RiskScore"].max())

selected_score_range = st.sidebar.slider(
    "Risk Score Range",
    min_value=round(min_score, 2),
    max_value=round(max_score, 2),
    value=(round(min_score, 2), round(max_score, 2)),
)

df_filtered = df[
    (df["RiskLevel"].astype(str).isin(selected_risk))
    & (df["TransactionType"].astype(str).isin(selected_transaction_type))
    & (df["Channel"].astype(str).isin(selected_channel))
    & (df["CustomerOccupation"].astype(str).isin(selected_occupation))
    & (df["Location"].astype(str).isin(selected_location))
    & (df["RiskScore"] >= selected_score_range[0])
    & (df["RiskScore"] <= selected_score_range[1])
].copy()

if len(df_filtered) == 0:
    st.warning("Tidak ada data yang sesuai dengan filter.")
    st.stop()


tab_overview, tab_transactions, tab_explainability, tab_data, tab_prediction = st.tabs(
    [
        "Overview",
        "High Risk Transactions",
        "Explainability",
        "Data Explorer",
        "Prediction",
    ]
)


with tab_overview:
    st.markdown('<div class="section-title">Filtered Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Ringkasan hasil anomaly detection dan risk scoring berdasarkan filter yang dipilih.</div>',
        unsafe_allow_html=True,
    )

    total_transactions = len(df_filtered)
    total_anomaly = int(df_filtered["Ensemble_Anomaly"].sum())
    anomaly_percentage = total_anomaly / total_transactions * 100 if total_transactions > 0 else 0
    high_risk_count = int((df_filtered["RiskLevel"].astype(str) == "High").sum())
    average_risk_score = df_filtered["RiskScore"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        metric_card("Total Transactions", format_number(total_transactions), "Filtered transaction count")

    with col2:
        metric_card("Anomaly Transactions", format_number(total_anomaly), "Ensemble anomaly = 1")

    with col3:
        metric_card("Anomaly Rate", format_percentage(anomaly_percentage), "Anomaly proportion")

    with col4:
        metric_card("High Risk", format_number(high_risk_count), "RiskLevel = High")

    with col5:
        metric_card("Avg Risk Score", f"{average_risk_score:.2f}", "Average score 0-100")

    st.write("")

    col_left, col_right = st.columns(2)

    with col_left:
        risk_distribution = (
            df_filtered["RiskLevel"]
            .astype(str)
            .value_counts()
            .reindex(RISK_ORDER)
            .fillna(0)
            .reset_index()
        )

        risk_distribution.columns = ["RiskLevel", "Count"]

        fig_risk = px.bar(
            risk_distribution,
            x="RiskLevel",
            y="Count",
            text="Count",
            color="RiskLevel",
            color_discrete_map=RISK_COLOR_MAP,
            title="Risk level distribution",
        )

        style_plotly_chart(fig_risk, height=360, showlegend=False)
        fig_risk.update_traces(textposition="outside", cliponaxis=False)

        st.plotly_chart(fig_risk, width="stretch")

    with col_right:
        vote_distribution = (
            df_filtered["AnomalyVoteCount"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        vote_distribution.columns = ["AnomalyVoteCount", "Count"]

        fig_vote = px.bar(
            vote_distribution,
            x="AnomalyVoteCount",
            y="Count",
            text="Count",
            title="Anomaly vote count distribution",
        )

        style_plotly_chart(fig_vote, height=360, showlegend=False)
        fig_vote.update_traces(marker_color=DEFAULT_BAR_COLOR, textposition="outside", cliponaxis=False)

        st.plotly_chart(fig_vote, width="stretch")

    col_left, col_right = st.columns(2)

    with col_left:
        risk_by_channel_df = df_filtered.copy()
        risk_by_channel_df["RiskLevelText"] = risk_by_channel_df["RiskLevel"].astype(str)

        risk_by_channel = (
            risk_by_channel_df
            .groupby(["Channel", "RiskLevelText"])
            .size()
            .reset_index(name="Count")
        )

        fig_channel = px.bar(
            risk_by_channel,
            x="Channel",
            y="Count",
            color="RiskLevelText",
            color_discrete_map=RISK_COLOR_MAP,
            barmode="group",
            title="Risk level by channel",
        )

        style_plotly_chart(fig_channel, height=360)

        st.plotly_chart(fig_channel, width="stretch")

    with col_right:
        risk_by_type_df = df_filtered.copy()
        risk_by_type_df["RiskLevelText"] = risk_by_type_df["RiskLevel"].astype(str)

        risk_by_type = (
            risk_by_type_df
            .groupby(["TransactionType", "RiskLevelText"])
            .size()
            .reset_index(name="Count")
        )

        fig_type = px.bar(
            risk_by_type,
            x="TransactionType",
            y="Count",
            color="RiskLevelText",
            color_discrete_map=RISK_COLOR_MAP,
            barmode="group",
            title="Risk level by transaction type",
        )

        style_plotly_chart(fig_type, height=360)

        st.plotly_chart(fig_type, width="stretch")

    st.markdown('<div class="section-title">High Risk by Location</div>', unsafe_allow_html=True)

    high_location = (
        df_filtered[df_filtered["RiskLevel"].astype(str) == "High"]
        .groupby("Location")
        .size()
        .reset_index(name="HighRiskCount")
        .sort_values("HighRiskCount", ascending=False)
        .head(10)
    )

    if len(high_location) > 0:
        fig_location = px.bar(
            high_location.sort_values("HighRiskCount", ascending=True),
            x="HighRiskCount",
            y="Location",
            orientation="h",
            text="HighRiskCount",
            title="Top 10 locations by high-risk count",
        )

        style_plotly_chart(fig_location, height=360, showlegend=False)
        fig_location.update_traces(marker_color=RISK_COLOR_MAP["High"], textposition="outside", cliponaxis=False)

        st.plotly_chart(fig_location, width="stretch")
    else:
        st.info("Tidak ada transaksi High Risk pada filter ini.")


with tab_transactions:
    st.markdown('<div class="section-title">Top High Risk Transactions</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Transaksi diurutkan berdasarkan RiskScore tertinggi.</div>',
        unsafe_allow_html=True,
    )

    top_n = st.slider(
        "Jumlah transaksi yang ditampilkan",
        min_value=5,
        max_value=100,
        value=20,
    )

    top_columns = [
        "TransactionID",
        "AccountID",
        "TransactionAmount",
        "AccountBalance",
        "Amount_to_Balance_Ratio",
        "Amount_to_TypeChannelBalanceGroupAvg_Ratio",
        "TransactionType",
        "Channel",
        "Location",
        "CustomerAge",
        "CustomerOccupation",
        "TransactionDuration",
        "IsLongDuration",
        "LoginAttempts",
        "IsHighLoginAttempts",
        "AnomalyVoteCount",
        "Ensemble_Anomaly",
        "RiskScore",
        "RiskLevel",
    ]

    existing_top_columns = safe_columns(df_filtered, top_columns)

    top_risk = (
        df_filtered
        .sort_values(["RiskScore", "AnomalyVoteCount"], ascending=False)
        .head(top_n)
    )

    st.dataframe(
        top_risk[existing_top_columns],
        width="stretch",
        height=520,
    )

    st.markdown('<div class="section-title">Transaction Detail</div>', unsafe_allow_html=True)

    transaction_ids = (
        df_filtered
        .sort_values("RiskScore", ascending=False)["TransactionID"]
        .astype(str)
        .tolist()
    )

    selected_transaction_id = st.selectbox(
        "Pilih TransactionID",
        options=transaction_ids,
    )

    selected_transaction = df_filtered[
        df_filtered["TransactionID"].astype(str) == selected_transaction_id
    ]

    if len(selected_transaction) > 0:
        selected_row = selected_transaction.iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            metric_card("Risk Score", f"{selected_row['RiskScore']:.2f}", "Score range 0-100")

        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Risk Level</div>
                    <div style="margin-top: 14px;">{risk_badge_html(selected_row["RiskLevel"])}</div>
                    <div class="metric-note">Risk category</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            metric_card("Anomaly Vote", f"{int(selected_row['AnomalyVoteCount'])} / 4", "Model agreement")

        with col4:
            metric_card("Ensemble Anomaly", f"{int(selected_row['Ensemble_Anomaly'])}", "1 means anomaly")

        st.write("")

        detail_left, detail_right = st.columns([1.2, 1])

        with detail_left:
            st.markdown('<div class="section-title">Transaction Profile</div>', unsafe_allow_html=True)

            detail_columns = [
                "TransactionID",
                "AccountID",
                "TransactionAmount",
                "AccountBalance",
                "Amount_to_Balance_Ratio",
                "Amount_to_TypeChannelBalanceGroupAvg_Ratio",
                "TransactionType",
                "Channel",
                "Location",
                "CustomerAge",
                "CustomerOccupation",
                "TransactionDuration",
                "IsLongDuration",
                "LoginAttempts",
                "IsHighLoginAttempts",
                "BalanceGroup_ByType",
                "AccountTxCount",
                "DeviceTxCount",
                "MerchantTxCount",
                "IPTxCount",
            ]

            existing_detail_columns = safe_columns(selected_transaction, detail_columns)

            detail_table = (
                selected_transaction[existing_detail_columns]
                .T
                .reset_index()
            )

            detail_table.columns = ["Feature", "Value"]
            detail_table = stringify_value_column(detail_table)

            st.dataframe(
                detail_table,
                width="stretch",
                height=520,
            )

        with detail_right:
            st.markdown('<div class="section-title">Model Votes</div>', unsafe_allow_html=True)

            model_vote_columns = [
                "IF_Anomaly",
                "LOF_Anomaly",
                "OCSVM_Anomaly",
                "MCD_Anomaly",
            ]

            model_vote_names = [
                "Isolation Forest",
                "LOF",
                "One-Class SVM",
                "MCD",
            ]

            existing_vote_columns = safe_columns(selected_transaction, model_vote_columns)

            if len(existing_vote_columns) > 0:
                vote_table = pd.DataFrame({
                    "Model": [
                        model_vote_names[model_vote_columns.index(col)]
                        for col in existing_vote_columns
                    ],
                    "Anomaly Flag": [
                        int(selected_row[col])
                        for col in existing_vote_columns
                    ],
                })

                fig_model_vote = px.bar(
                    vote_table,
                    x="Model",
                    y="Anomaly Flag",
                    text="Anomaly Flag",
                    title="Anomaly vote by model",
                )

                style_plotly_chart(fig_model_vote, height=300, showlegend=False)
                fig_model_vote.update_traces(marker_color=DEFAULT_BAR_COLOR)
                fig_model_vote.update_yaxes(range=[0, 1.2])

                st.plotly_chart(fig_model_vote, width="stretch")

            shap_columns = [
                "Top_SHAP_Feature_1",
                "Top_SHAP_Value_1",
                "Top_SHAP_Feature_2",
                "Top_SHAP_Value_2",
                "Top_SHAP_Feature_3",
                "Top_SHAP_Value_3",
            ]

            existing_shap_columns = safe_columns(selected_transaction, shap_columns)

            if len(existing_shap_columns) == len(shap_columns):
                st.markdown('<div class="section-title">Top SHAP Drivers</div>', unsafe_allow_html=True)

                shap_driver_table = pd.DataFrame({
                    "Top Feature": [
                        selected_row["Top_SHAP_Feature_1"],
                        selected_row["Top_SHAP_Feature_2"],
                        selected_row["Top_SHAP_Feature_3"],
                    ],
                    "SHAP Value": [
                        selected_row["Top_SHAP_Value_1"],
                        selected_row["Top_SHAP_Value_2"],
                        selected_row["Top_SHAP_Value_3"],
                    ],
                })

                st.dataframe(
                    shap_driver_table,
                    width="stretch",
                )
            else:
                st.info("Top SHAP feature per transaksi belum tersedia di file final result.")


with tab_explainability:
    st.markdown('<div class="section-title">Explainability</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box">
            XGBoost digunakan sebagai surrogate model untuk menjelaskan hasil ensemble anomaly detection.
            Model ini tidak dilatih menggunakan label fraud asli, tetapi menggunakan Ensemble_Anomaly sebagai pseudo-label.
            Feature importance menunjukkan fitur yang paling membantu surrogate model dalam membedakan transaksi normal dan anomali.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    if len(feature_importance) > 0 and {"Feature", "Importance"}.issubset(feature_importance.columns):
        top_n_importance = st.slider(
            "Jumlah feature importance yang ditampilkan",
            min_value=10,
            max_value=min(50, len(feature_importance)),
            value=min(20, len(feature_importance)),
            key="feature_importance_slider",
        )

        top_feature_importance = (
            feature_importance
            .head(top_n_importance)
            .sort_values("Importance", ascending=True)
        )

        fig_importance = px.bar(
            top_feature_importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Top {top_n_importance} feature importance",
        )

        style_plotly_chart(fig_importance, height=560, showlegend=False)
        fig_importance.update_traces(marker_color=DEFAULT_BAR_COLOR)

        st.plotly_chart(fig_importance, width="stretch")

        st.dataframe(
            feature_importance.head(50),
            width="stretch",
            height=520,
        )
    else:
        st.info("File feature importance belum tersedia atau format kolomnya belum sesuai.")

    st.markdown('<div class="section-title">Risk Pattern Summary</div>', unsafe_allow_html=True)

    pattern_columns = safe_columns(
        df_filtered,
        [
            "RiskLevel",
            "TransactionAmount",
            "AccountBalance",
            "Amount_to_Balance_Ratio",
            "Amount_to_TypeChannelBalanceGroupAvg_Ratio",
            "TransactionDuration",
            "LoginAttempts",
            "AnomalyVoteCount",
            "RiskScore",
        ],
    )

    if len(pattern_columns) > 1:
        pattern_summary = (
            df_filtered[pattern_columns]
            .groupby("RiskLevel")
            .mean(numeric_only=True)
            .reindex(RISK_ORDER)
            .dropna(how="all")
            .round(3)
        )

        st.dataframe(
            pattern_summary,
            width="stretch",
        )


with tab_data:
    st.markdown('<div class="section-title">Data Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Gunakan tab ini untuk melihat data hasil akhir setelah anomaly detection, risk scoring, dan feature engineering.</div>',
        unsafe_allow_html=True,
    )

    st.write(f"Filtered rows: {len(df_filtered):,}")

    default_data_columns = safe_columns(
        df_filtered,
        [
            "TransactionID",
            "AccountID",
            "TransactionAmount",
            "AccountBalance",
            "TransactionType",
            "Channel",
            "Location",
            "AnomalyVoteCount",
            "Ensemble_Anomaly",
            "RiskScore",
            "RiskLevel",
        ],
    )

    selected_columns = st.multiselect(
        "Pilih kolom yang ingin ditampilkan",
        options=df_filtered.columns.tolist(),
        default=default_data_columns,
    )

    if len(selected_columns) > 0:
        st.dataframe(
            df_filtered[selected_columns],
            width="stretch",
            height=650,
        )

        csv_data = df_filtered[selected_columns].to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv_data,
            file_name="filtered_bank_transaction_anomaly_result.csv",
            mime="text/csv",
        )
    else:
        st.warning("Pilih minimal satu kolom.")


with tab_prediction:
    st.markdown('<div class="section-title">New Transaction Prediction</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box">
            Halaman ini digunakan untuk mengecek apakah transaksi baru memiliki pola normal atau anomali
            berdasarkan model anomaly detection yang sudah dilatih. Hasil prediksi berupa risk score dan risk level,
            bukan label fraud aktual.
        </div>
        """,
        unsafe_allow_html=True,
    )

    prediction_bundle, missing_prediction_files = load_prediction_artifacts()

    if prediction_bundle is None:
        st.error("Beberapa artifact prediction belum ditemukan.")
        st.write("File yang belum ada:")
        st.write(missing_prediction_files)
    else:
        st.write("")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown('<div class="subsection-title">Transaction Information</div>', unsafe_allow_html=True)

            transaction_amount = st.number_input(
                "Transaction Amount",
                min_value=0.0,
                value=1000.0,
                step=10.0,
            )

            account_balance = st.number_input(
                "Account Balance",
                min_value=0.0,
                value=5000.0,
                step=100.0,
            )

            transaction_duration = st.number_input(
                "Transaction Duration",
                min_value=1,
                value=120,
                step=1,
            )

            login_attempts = st.number_input(
                "Login Attempts",
                min_value=1,
                value=1,
                step=1,
            )

            customer_age = st.number_input(
                "Customer Age",
                min_value=1,
                max_value=100,
                value=35,
                step=1,
            )

            transaction_date = st.date_input("Transaction Date")
            transaction_time = st.time_input("Transaction Time")

        with col_right:
            st.markdown('<div class="subsection-title">Transaction Context</div>', unsafe_allow_html=True)

            transaction_type = st.selectbox(
                "Transaction Type",
                options=sorted(df["TransactionType"].dropna().astype(str).unique().tolist()),
            )

            channel = st.selectbox(
                "Channel",
                options=sorted(df["Channel"].dropna().astype(str).unique().tolist()),
            )

            location = st.selectbox(
                "Location",
                options=sorted(df["Location"].dropna().astype(str).unique().tolist()),
            )

            customer_occupation = st.selectbox(
                "Customer Occupation",
                options=sorted(df["CustomerOccupation"].dropna().astype(str).unique().tolist()),
            )

            account_id = st.text_input("Account ID", value="NEW_ACCOUNT")
            device_id = st.text_input("Device ID", value="NEW_DEVICE")
            merchant_id = st.text_input("Merchant ID", value="NEW_MERCHANT")
            ip_address = st.text_input("IP Address", value="0.0.0.0")

        input_data = {
            "TransactionAmount": transaction_amount,
            "AccountBalance": account_balance,
            "TransactionDuration": transaction_duration,
            "LoginAttempts": login_attempts,
            "CustomerAge": customer_age,
            "TransactionDate": f"{transaction_date} {transaction_time}",
            "TransactionType": transaction_type,
            "Channel": channel,
            "Location": location,
            "CustomerOccupation": customer_occupation,
            "AccountID": account_id,
            "DeviceID": device_id,
            "MerchantID": merchant_id,
            "IP Address": ip_address,
        }

        st.write("")

        predict_button = st.button("Predict Transaction Risk", width="stretch")

        if predict_button:
            df_new_raw, prediction_result = predict_new_transaction(
                input_data,
                prediction_bundle,
            )

            st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                metric_card("Risk Score", f"{prediction_result['RiskScore']:.2f}", "Score range 0-100")

            with col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Risk Level</div>
                        <div style="margin-top: 14px;">{risk_badge_html(prediction_result["RiskLevel"])}</div>
                        <div class="metric-note">Risk category</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col3:
                metric_card("Anomaly Vote", f"{prediction_result['AnomalyVoteCount']} / 4", "Model agreement")

            with col4:
                status_text = "Anomaly" if prediction_result["Ensemble_Anomaly"] == 1 else "Normal"
                metric_card("Prediction Status", status_text, "Based on ensemble voting")

            st.write("")

            vote_table = pd.DataFrame({
                "Model": [
                    "Isolation Forest",
                    "LOF",
                    "One-Class SVM",
                    "MCD",
                ],
                "Anomaly Flag": [
                    prediction_result["IF_Anomaly"],
                    prediction_result["LOF_Anomaly"],
                    prediction_result["OCSVM_Anomaly"],
                    prediction_result["MCD_Anomaly"],
                ],
            })

            col_left, col_right = st.columns([1, 1])

            with col_left:
                fig_vote_prediction = px.bar(
                    vote_table,
                    x="Model",
                    y="Anomaly Flag",
                    text="Anomaly Flag",
                    title="Model votes for new transaction",
                )

                style_plotly_chart(fig_vote_prediction, height=380, showlegend=False)
                fig_vote_prediction.update_traces(marker_color=DEFAULT_BAR_COLOR)
                fig_vote_prediction.update_yaxes(range=[0, 1.2])

                st.plotly_chart(fig_vote_prediction, width="stretch")

            with col_right:
                result_table = pd.DataFrame({
                    "Metric": [
                        "IF_Anomaly",
                        "LOF_Anomaly",
                        "OCSVM_Anomaly",
                        "MCD_Anomaly",
                        "AnomalyVoteCount",
                        "Ensemble_Anomaly",
                        "Average_AnomalyScore_Norm",
                        "RiskScore",
                        "RiskLevel",
                    ],
                    "Value": [
                        prediction_result["IF_Anomaly"],
                        prediction_result["LOF_Anomaly"],
                        prediction_result["OCSVM_Anomaly"],
                        prediction_result["MCD_Anomaly"],
                        prediction_result["AnomalyVoteCount"],
                        prediction_result["Ensemble_Anomaly"],
                        round(prediction_result["Average_AnomalyScore_Norm"], 4),
                        prediction_result["RiskScore"],
                        prediction_result["RiskLevel"],
                    ],
                })
                result_table = stringify_value_column(result_table)

                st.dataframe(
                    result_table,
                    width="stretch",
                    height=420,
                )

            st.markdown('<div class="section-title">Generated Features</div>', unsafe_allow_html=True)

            generated_feature_columns = [
                "TransactionAmount",
                "AccountBalance",
                "Amount_to_Balance_Ratio",
                "Amount_to_TypeChannelBalanceGroupAvg_Ratio",
                "TransactionType",
                "Channel",
                "Location",
                "CustomerAge",
                "CustomerOccupation",
                "TransactionDuration",
                "IsLongDuration",
                "LoginAttempts",
                "IsHighLoginAttempts",
                "BalanceGroup_ByType",
                "AccountTxCount",
                "DeviceTxCount",
                "MerchantTxCount",
                "IPTxCount",
            ]

            existing_generated_columns = safe_columns(df_new_raw, generated_feature_columns)

            generated_table = (
                df_new_raw[existing_generated_columns]
                .T
                .reset_index()
            )

            generated_table.columns = ["Feature", "Value"]
            generated_table = stringify_value_column(generated_table)

            st.dataframe(
                generated_table,
                width="stretch",
            )
