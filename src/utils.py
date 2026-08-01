import pandas as pd


FEATURE_LABELS = {
    "TransactionID": "ID transaksi",
    "AccountID": "ID rekening",
    "DeviceID": "ID perangkat",
    "MerchantID": "ID merchant",
    "IP Address": "Alamat IP",
    "TransactionAmount": "Nilai transaksi",
    "AccountBalance": "Saldo rekening",
    "TransactionType": "Tipe transaksi",
    "Channel": "Kanal",
    "Location": "Lokasi",
    "CustomerOccupation": "Pekerjaan nasabah",
    "Amount_to_Balance_Ratio": "Rasio transaksi terhadap saldo",
    "Amount_to_TypeChannelBalanceGroupAvg_Ratio": "Rasio terhadap rata-rata kelompok",
    "TransactionDuration": "Durasi transaksi",
    "LoginAttempts": "Percobaan login",
    "AccountTxCount": "Frekuensi rekening",
    "DeviceTxCount": "Frekuensi perangkat",
    "MerchantTxCount": "Frekuensi merchant",
    "IPTxCount": "Frekuensi alamat IP",
    "IsHighLoginAttempts": "Percobaan login tinggi",
    "IsLongDuration": "Durasi panjang",
    "CustomerAge": "Usia nasabah",
    "AnomalyVoteCount": "Jumlah model yang menandai",
    "Ensemble_Anomaly": "Keputusan ensemble",
    "RiskScore": "Skor risiko",
    "RiskLevel": "Level risiko",
    "VoteScore": "Skor voting",
    "Average_AnomalyScore_Norm": "Rata-rata skor anomali normalisasi",
    "IF_Anomaly": "Keputusan Isolation Forest",
    "LOF_Anomaly": "Keputusan Local Outlier Factor",
    "OCSVM_Anomaly": "Keputusan One-Class SVM",
    "MCD_Anomaly": "Keputusan MCD",
    "Hour": "Jam",
    "DayOfWeek": "Hari dalam minggu",
    "Month": "Bulan",
    "Hour_sin": "Pola jam (sin)",
    "Hour_cos": "Pola jam (cos)",
    "DayOfWeek_sin": "Pola hari (sin)",
    "DayOfWeek_cos": "Pola hari (cos)",
    "Month_sin": "Pola bulan (sin)",
    "Month_cos": "Pola bulan (cos)",
}

RISK_LABELS_ID = {
    "Low": "Rendah",
    "Low-Medium": "Rendah–sedang",
    "Medium": "Sedang",
    "High": "Tinggi",
}


def humanize_feature_name(value):
    text = str(value)

    if text in FEATURE_LABELS:
        return FEATURE_LABELS[text]

    for prefix, label in (
        ("TransactionType_", "Tipe transaksi: "),
        ("Channel_", "Kanal: "),
        ("Location_", "Lokasi: "),
        ("CustomerOccupation_", "Pekerjaan: "),
        ("BalanceGroup_ByType_", "Kelompok saldo: "),
    ):
        if text.startswith(prefix):
            return f"{label}{text.removeprefix(prefix)}"

    return text.replace("_", " ")


def translate_risk_level(value):
    return RISK_LABELS_ID.get(str(value), "Tidak diketahui")


def safe_columns(dataframe, columns):
    return [column for column in columns if column in dataframe.columns]


def stringify_value_column(table, value_column="Value"):
    table = table.copy()

    if value_column in table.columns:
        table[value_column] = table[value_column].apply(
            lambda value: "" if pd.isna(value) else str(value)
        )

    return table
