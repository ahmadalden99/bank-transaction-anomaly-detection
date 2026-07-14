import pandas as pd
import streamlit as st

from dashboard.components.layout import info_box, section_title
from src.utils import humanize_feature_name, safe_columns, translate_risk_level


COLUMN_PRESETS = {
    "Ringkas": [
        "TransactionID",
        "AccountID",
        "TransactionAmount",
        "TransactionType",
        "Channel",
        "Location",
        "AnomalyVoteCount",
        "RiskScore",
        "RiskLevel",
    ],
    "Tinjauan risiko": [
        "TransactionID",
        "AccountID",
        "TransactionAmount",
        "AccountBalance",
        "Amount_to_Balance_Ratio",
        "TransactionType",
        "Channel",
        "Location",
        "LoginAttempts",
        "TransactionDuration",
        "IF_Anomaly",
        "LOF_Anomaly",
        "OCSVM_Anomaly",
        "MCD_Anomaly",
        "AnomalyVoteCount",
        "Ensemble_Anomaly",
        "RiskScore",
        "RiskLevel",
    ],
    "Semua kolom": [],
}


def render_data_explorer_tab(dataframe):
    section_title(
        "Jelajahi data transaksi",
        "Cari transaksi, pilih kolom, dan unduh hasil pada cakupan aktif.",
    )

    control_left, control_right = st.columns([1.2, 1])
    with control_left:
        search_query = st.text_input(
            "Cari data",
            placeholder="Cari ID transaksi, rekening, lokasi, kanal…",
            key="data_search",
        )
    with control_right:
        preset = st.radio(
            "Tampilan kolom",
            options=list(COLUMN_PRESETS),
            horizontal=True,
            key="data_preset",
        )

    searched_dataframe = _search_dataframe(dataframe, search_query)
    preset_columns = (
        dataframe.columns.tolist()
        if preset == "Semua kolom"
        else safe_columns(dataframe, COLUMN_PRESETS[preset])
    )

    with st.expander("Sesuaikan kolom secara manual"):
        selected_columns = st.multiselect(
            "Kolom yang ditampilkan",
            options=dataframe.columns.tolist(),
            format_func=humanize_feature_name,
            default=preset_columns,
            key=f"data_columns_{preset}",
        )

    if not selected_columns:
        info_box("Pilih minimal satu kolom untuk menampilkan tabel.", tone="warning")
        return

    st.caption(
        f"{len(searched_dataframe):,} baris · {len(selected_columns):,} kolom dipilih · "
        f"cakupan awal {len(dataframe):,} baris"
    )

    if searched_dataframe.empty:
        info_box(
            "Pencarian tidak menemukan transaksi. Coba kata kunci yang lebih pendek.",
            tone="warning",
        )
        return

    display_dataframe = searched_dataframe[selected_columns].copy()
    if "RiskLevel" in display_dataframe.columns:
        display_dataframe["RiskLevel"] = display_dataframe["RiskLevel"].map(
            translate_risk_level
        )

    st.dataframe(
        display_dataframe,
        width="stretch",
        height=620,
        hide_index=True,
        column_config={
            "TransactionID": st.column_config.TextColumn("ID transaksi", pinned=True),
            "TransactionAmount": st.column_config.NumberColumn("Nilai transaksi", format="%.2f"),
            "AccountBalance": st.column_config.NumberColumn("Saldo rekening", format="%.2f"),
            "RiskScore": st.column_config.NumberColumn("Skor risiko", format="%.2f"),
            "AnomalyVoteCount": st.column_config.NumberColumn("Model menandai", format="%d dari 4"),
            "RiskLevel": st.column_config.TextColumn("Level risiko"),
        },
    )

    export_data = searched_dataframe[selected_columns].to_csv(index=False).encode("utf-8")
    download_left, download_right = st.columns([1, 2.2])
    with download_left:
        st.download_button(
            label="Unduh CSV tampilan ini",
            data=export_data,
            file_name="hasil_deteksi_anomali.csv",
            mime="text/csv",
            width="stretch",
            on_click="ignore",
        )
    with download_right:
        st.caption(
            "File mengikuti filter sidebar, pencarian, dan kolom yang dipilih pada halaman ini."
        )


def _search_dataframe(dataframe, search_query):
    if not search_query.strip():
        return dataframe

    searchable_columns = safe_columns(
        dataframe,
        [
            "TransactionID",
            "AccountID",
            "Location",
            "Channel",
            "TransactionType",
            "CustomerOccupation",
        ],
    )
    search_mask = pd.Series(False, index=dataframe.index)
    for column in searchable_columns:
        search_mask |= dataframe[column].astype(str).str.contains(
            search_query.strip(),
            case=False,
            na=False,
            regex=False,
        )

    return dataframe.loc[search_mask].copy()
