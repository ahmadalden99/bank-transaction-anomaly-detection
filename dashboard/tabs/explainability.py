import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.charts import style_plotly_chart
from dashboard.components.layout import info_box, section_title
from src.config import DEFAULT_BAR_COLOR, RISK_ORDER
from src.utils import humanize_feature_name, safe_columns, translate_risk_level


def render_explainability_tab(dataframe, feature_importance):
    info_box(
        "XGBoost digunakan sebagai model surrogate untuk membantu menjelaskan keputusan ensemble. "
        "Model ini mempelajari hasil deteksi anomali, bukan label fraud aktual.",
        tone="neutral",
    )

    if (
        feature_importance.empty
        or not {"Feature", "Importance"}.issubset(feature_importance.columns)
    ):
        info_box("Data kepentingan fitur belum tersedia.", tone="warning")
    else:
        _render_feature_importance(feature_importance)

    _render_risk_patterns(dataframe)


def _render_feature_importance(feature_importance):
    section_title(
        "Fitur yang paling berpengaruh",
        "Nilai ini menunjukkan kontribusi relatif fitur pada model surrogate untuk seluruh dataset.",
    )
    importance = feature_importance.copy()
    importance["Importance"] = pd.to_numeric(
        importance["Importance"],
        errors="coerce",
    ).fillna(0)
    importance = importance.sort_values("Importance", ascending=False).reset_index(drop=True)
    importance["Label"] = importance["Feature"].map(humanize_feature_name)

    maximum = min(30, len(importance))
    minimum = min(5, maximum)
    if maximum > minimum:
        top_n = st.slider(
            "Jumlah fitur pada grafik",
            min_value=minimum,
            max_value=maximum,
            value=min(15, maximum),
            key="feature_importance_slider",
        )
    else:
        top_n = maximum

    chart_data = importance.head(top_n).sort_values("Importance")
    fig = px.bar(
        chart_data,
        x="Importance",
        y="Label",
        orientation="h",
        title=f"{top_n} fitur dengan tingkat kepentingan tertinggi",
        labels={"Importance": "Tingkat kepentingan", "Label": "Fitur"},
        custom_data=["Feature"],
    )
    style_plotly_chart(fig, height=max(390, top_n * 28), showlegend=False)
    fig.update_traces(
        marker_color=DEFAULT_BAR_COLOR,
        hovertemplate=(
            "%{y}<br>Tingkat kepentingan: %{x:.4f}<br>Nama teknis: %{customdata[0]}<extra></extra>"
        ),
    )
    st.plotly_chart(fig, width="stretch")

    with st.expander("Lihat seluruh nilai kepentingan fitur"):
        st.dataframe(
            importance[["Label", "Feature", "Importance"]],
            width="stretch",
            height=450,
            hide_index=True,
            column_config={
                "Label": st.column_config.TextColumn("Fitur", width="large"),
                "Feature": st.column_config.TextColumn("Nama teknis", width="large"),
                "Importance": st.column_config.NumberColumn(
                    "Tingkat kepentingan",
                    format="%.4f",
                ),
            },
        )


def _render_risk_patterns(dataframe):
    section_title(
        "Perbedaan pola antar-level risiko",
        "Rata-rata nilai pada data yang sedang ditampilkan. Perbedaan ini tidak menunjukkan hubungan sebab-akibat.",
    )
    columns = safe_columns(
        dataframe,
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
    if "RiskLevel" not in columns or len(columns) <= 1:
        st.caption("Kolom untuk membuat ringkasan pola belum cukup.")
        return

    summary = (
        dataframe[columns]
        .groupby("RiskLevel", observed=True)
        .mean(numeric_only=True)
        .reindex(RISK_ORDER)
        .dropna(how="all")
        .round(3)
        .reset_index()
    )
    summary["RiskLevel"] = summary["RiskLevel"].map(translate_risk_level)
    summary.columns = [
        "Level risiko" if column == "RiskLevel" else humanize_feature_name(column)
        for column in summary.columns
    ]

    st.dataframe(
        summary,
        width="stretch",
        hide_index=True,
        column_config={
            "Level risiko": st.column_config.TextColumn("Level risiko", pinned=True),
            "Nilai transaksi": st.column_config.NumberColumn("Nilai transaksi", format="%.2f"),
            "Saldo rekening": st.column_config.NumberColumn("Saldo rekening", format="%.2f"),
            "Skor risiko": st.column_config.NumberColumn("Skor risiko", format="%.2f"),
        },
    )
