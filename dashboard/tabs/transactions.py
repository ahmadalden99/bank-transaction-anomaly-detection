from html import escape

import pandas as pd
import streamlit as st

from dashboard.components.cards import metric_card, risk_metric_card
from dashboard.components.layout import info_box, section_title, subsection_title
from src.utils import humanize_feature_name, safe_columns, translate_risk_level


QUEUE_COLUMNS = [
    "TransactionID",
    "AccountID",
    "TransactionAmount",
    "TransactionType",
    "Channel",
    "Location",
    "AnomalyVoteCount",
    "RiskScore",
    "RiskLevel",
]

MODEL_VOTE_COLUMNS = {
    "IF_Anomaly": "Isolation Forest",
    "LOF_Anomaly": "Local Outlier Factor",
    "OCSVM_Anomaly": "One-Class SVM",
    "MCD_Anomaly": "MCD / Elliptic Envelope",
}

PROFILE_FIELDS = [
    ("TransactionAmount", "Nilai transaksi"),
    ("AccountBalance", "Saldo rekening"),
    ("TransactionType", "Tipe transaksi"),
    ("Channel", "Kanal"),
    ("Location", "Lokasi"),
    ("CustomerAge", "Usia nasabah"),
    ("CustomerOccupation", "Pekerjaan nasabah"),
    ("TransactionDuration", "Durasi transaksi"),
    ("LoginAttempts", "Percobaan login"),
    ("BalanceGroup_ByType", "Kelompok saldo"),
]

TECHNICAL_FIELDS = [
    "Amount_to_Balance_Ratio",
    "Amount_to_TypeChannelBalanceGroupAvg_Ratio",
    "IsLongDuration",
    "IsHighLoginAttempts",
    "AccountTxCount",
    "DeviceTxCount",
    "MerchantTxCount",
    "IPTxCount",
]


def render_transactions_tab(dataframe):
    section_title(
        "Pilih transaksi untuk ditinjau",
        "Urutan awal berdasarkan skor risiko dan jumlah model yang menandai transaksi.",
    )

    left, middle, right = st.columns([1.15, 1.25, 0.7])
    with left:
        queue_scope = st.radio(
            "Cakupan antrian",
            options=["Prioritas", "Anomali ensemble", "Semua"],
            horizontal=True,
            key="queue_scope",
        )
    with middle:
        search_query = st.text_input(
            "Cari transaksi",
            placeholder="ID transaksi, rekening, atau lokasi",
            key="queue_search",
        )
    with right:
        row_limit = st.selectbox(
            "Jumlah baris",
            options=[25, 50, 100, 250],
            index=1,
            key="queue_limit",
        )

    queue = _build_queue(dataframe, queue_scope, search_query).head(row_limit)
    if queue.empty:
        info_box("Tidak ada transaksi yang cocok dengan pilihan ini.", tone="warning")
        return

    display = queue[safe_columns(queue, QUEUE_COLUMNS)].copy()
    display["RiskLevel"] = display["RiskLevel"].map(translate_risk_level)

    event = st.dataframe(
        display,
        width="stretch",
        height=min(510, 82 + len(queue) * 35),
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="risk_queue",
        column_config={
            "TransactionID": st.column_config.TextColumn("ID transaksi", width="medium"),
            "AccountID": st.column_config.TextColumn("Rekening", width="small"),
            "TransactionAmount": st.column_config.NumberColumn("Nilai transaksi", format="%.2f"),
            "TransactionType": st.column_config.TextColumn("Tipe"),
            "Channel": st.column_config.TextColumn("Kanal"),
            "Location": st.column_config.TextColumn("Lokasi"),
            "AnomalyVoteCount": st.column_config.NumberColumn("Model menandai", format="%d dari 4"),
            "RiskScore": st.column_config.NumberColumn("Skor risiko", format="%.2f"),
            "RiskLevel": st.column_config.TextColumn("Level risiko"),
        },
    )

    if not event.selection.rows:
        info_box(
            "Pilih satu baris pada tabel untuk membuka detail transaksi.",
            tone="neutral",
        )
        return

    selected_row = queue.iloc[event.selection.rows[0]]
    _render_transaction_detail(selected_row)


def _build_queue(dataframe, scope, query):
    queue = dataframe.copy()
    if scope == "Prioritas":
        priority = queue[
            queue["RiskLevel"].astype(str).eq("High")
            | queue["Ensemble_Anomaly"].eq(1)
        ]
        if not priority.empty:
            queue = priority
    elif scope == "Anomali ensemble":
        queue = queue[queue["Ensemble_Anomaly"].eq(1)]

    if query.strip():
        mask = pd.Series(False, index=queue.index)
        for column in safe_columns(queue, ["TransactionID", "AccountID", "Location"]):
            mask |= queue[column].astype(str).str.contains(
                query.strip(),
                case=False,
                na=False,
                regex=False,
            )
        queue = queue.loc[mask]

    return queue.sort_values(
        ["RiskScore", "AnomalyVoteCount"],
        ascending=False,
    ).reset_index(drop=True)


def _render_transaction_detail(row):
    section_title(
        f"Detail transaksi {row['TransactionID']}",
        "Ringkasan hasil, informasi transaksi, dan bukti pendukung.",
    )

    columns = st.columns(4)
    with columns[0]:
        metric_card("Skor risiko", f"{row['RiskScore']:.2f} / 100", "Skor gabungan model")
    with columns[1]:
        risk_metric_card("Level risiko", row["RiskLevel"])
    with columns[2]:
        metric_card(
            "Model menandai",
            f"{int(row['AnomalyVoteCount'])} dari 4",
            "Keputusan ensemble",
        )
    with columns[3]:
        decision = "Tinjau" if int(row["Ensemble_Anomaly"]) else "Pantau"
        metric_card("Tindakan awal", decision, "Bukan konfirmasi fraud")

    _render_recommendation(row)

    left, right = st.columns([1.05, 1])
    with left:
        subsection_title("Informasi transaksi")
        _render_definition_list(row, PROFILE_FIELDS)
    with right:
        subsection_title("Keputusan model")
        _render_model_decisions(row)
        subsection_title("Faktor utama")
        _render_shap_drivers(row)

    with st.expander("Lihat fitur teknis"):
        technical_fields = [field for field in TECHNICAL_FIELDS if field in row.index]
        technical_pairs = [(field, humanize_feature_name(field)) for field in technical_fields]
        _render_definition_list(row, technical_pairs)


def _render_recommendation(row):
    if str(row["RiskLevel"]) == "High":
        info_box(
            "<strong>Saran:</strong> prioritaskan tinjauan manual dan verifikasi konteks rekening, perangkat, serta transaksi.",
            tone="danger",
        )
    elif int(row["Ensemble_Anomaly"]) == 1:
        info_box(
            "<strong>Saran:</strong> bandingkan transaksi dengan pola historis rekening sebelum mengambil tindakan.",
            tone="warning",
        )
    else:
        info_box(
            "<strong>Saran:</strong> belum diperlukan eskalasi otomatis; lanjutkan pemantauan sesuai prosedur.",
            tone="neutral",
        )


def _render_definition_list(row, fields):
    items = []
    for field, label in fields:
        if field not in row.index:
            continue
        value = _format_value(field, row[field])
        items.append(
            f"<div class='detail-row'><dt>{escape(label)}</dt><dd>{escape(value)}</dd></div>"
        )
    st.markdown(
        f"<dl class='detail-list'>{''.join(items)}</dl>",
        unsafe_allow_html=True,
    )


def _format_value(field, value):
    if pd.isna(value):
        return "—"
    if field in {"TransactionAmount", "AccountBalance"}:
        return f"{float(value):,.2f}"
    if field == "TransactionDuration":
        return f"{int(value)} detik"
    if isinstance(value, float):
        return f"{value:,.4f}"
    return str(value)


def _render_model_decisions(row):
    decisions = pd.DataFrame(
        {
            "Model": list(MODEL_VOTE_COLUMNS.values()),
            "Keputusan": [
                "Ditandai" if int(row[column]) else "Tidak ditandai"
                for column in MODEL_VOTE_COLUMNS
            ],
        }
    )
    st.dataframe(
        decisions,
        width="stretch",
        hide_index=True,
        column_config={
            "Model": st.column_config.TextColumn("Model", width="large"),
            "Keputusan": st.column_config.TextColumn("Keputusan", width="medium"),
        },
    )


def _render_shap_drivers(row):
    required = [
        "Top_SHAP_Feature_1",
        "Top_SHAP_Value_1",
        "Top_SHAP_Feature_2",
        "Top_SHAP_Value_2",
        "Top_SHAP_Feature_3",
        "Top_SHAP_Value_3",
    ]
    if not all(column in row.index for column in required):
        st.caption("Faktor SHAP per transaksi belum tersedia.")
        return

    drivers = []
    for index in range(1, 4):
        value = float(row[f"Top_SHAP_Value_{index}"])
        drivers.append(
            {
                "Fitur": humanize_feature_name(row[f"Top_SHAP_Feature_{index}"]),
                "Arah": "Ke anomali" if value > 0 else "Ke normal",
                "Nilai SHAP": value,
            }
        )
    st.dataframe(
        pd.DataFrame(drivers),
        width="stretch",
        hide_index=True,
        column_config={
            "Fitur": st.column_config.TextColumn("Fitur", width="large"),
            "Arah": st.column_config.TextColumn("Arah"),
            "Nilai SHAP": st.column_config.NumberColumn("Nilai SHAP", format="%.4f"),
        },
    )
