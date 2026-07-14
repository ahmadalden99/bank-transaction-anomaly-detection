import joblib
import numpy as np
import pandas as pd

from src.config import (
    IF_MODEL_PATH,
    LOF_MODEL_PATH,
    MCD_MODEL_PATH,
    NUMERIC_SCALER_PATH,
    OCSVM_MODEL_PATH,
    PREPROCESSING_ARTIFACTS_PATH,
    REQUIRED_PREDICTION_ARTIFACTS,
    RISK_SCORE_SCALER_PATH,
)


def load_prediction_artifacts():
    missing_paths = [
        str(path)
        for path in REQUIRED_PREDICTION_ARTIFACTS
        if not path.exists()
    ]

    if missing_paths:
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


def assign_balance_group(transaction_type, account_balance, balance_thresholds):
    if transaction_type in balance_thresholds.index:
        low_boundary = balance_thresholds.loc[
            transaction_type,
            "Low_to_Medium_Boundary",
        ]
        high_boundary = balance_thresholds.loc[
            transaction_type,
            "Medium_to_High_Boundary",
        ]

        if account_balance <= low_boundary:
            return "Low"

        if account_balance <= high_boundary:
            return "Medium"

        return "High"

    return "Medium"


def get_group_average_amount(
    transaction_type,
    channel,
    balance_group,
    avg_amount_by_group,
):
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

    if score >= 50:
        return "Medium"

    if score >= 25:
        return "Low-Medium"

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
    amount_to_balance_ratio = (
        transaction_amount / account_balance
        if account_balance != 0
        else 0
    )

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

    amount_to_group_avg_ratio = (
        transaction_amount / avg_amount
        if avg_amount != 0
        else 0
    )

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

    new_raw = pd.DataFrame([raw_row])
    new_numeric_raw = new_raw[numeric_features].copy()
    new_categorical_raw = new_raw[categorical_features].copy()

    new_numeric_scaled = pd.DataFrame(
        models["numeric_scaler"].transform(new_numeric_raw),
        columns=numeric_features,
    )

    new_categorical_encoded = pd.get_dummies(
        new_categorical_raw,
        columns=categorical_features,
        drop_first=False,
        dtype=int,
    )

    model_input = pd.concat(
        [
            new_numeric_scaled,
            new_categorical_encoded,
        ],
        axis=1,
    )

    model_input = model_input.reindex(columns=model_columns, fill_value=0)
    model_input = model_input.astype(float)

    return new_raw, model_input


def predict_new_transaction(input_data, prediction_bundle):
    models = prediction_bundle["models"]

    new_raw, model_input = build_new_transaction_features(input_data, prediction_bundle)

    if_pred = models["if_model"].predict(model_input)[0]
    lof_pred = models["lof_model"].predict(model_input)[0]
    ocsvm_pred = models["ocsvm_model"].predict(model_input)[0]
    mcd_pred = models["mcd_model"].predict(model_input)[0]

    if_score = -models["if_model"].decision_function(model_input)[0]
    lof_score = -models["lof_model"].decision_function(model_input)[0]
    ocsvm_score = -models["ocsvm_model"].decision_function(model_input)[0]
    mcd_score = -models["mcd_model"].decision_function(model_input)[0]

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

    risk_score = ((0.7 * vote_score) + (0.3 * average_anomaly_score_norm)) * 100
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

    return new_raw, prediction_result
