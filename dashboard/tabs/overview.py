import plotly.express as px
import streamlit as st

from dashboard.components.charts import style_plotly_chart
from dashboard.components.layout import section_title
from src.config import DEFAULT_BAR_COLOR, RISK_COLOR_MAP, RISK_ORDER
from src.utils import safe_columns, translate_risk_level


PRIORITY_COLUMNS = [
    "TransactionID",
    "TransactionAmount",
    "TransactionType",
    "Channel",
    "Location",
    "AnomalyVoteCount",
    "RiskScore",
    "RiskLevel",
]


def render_overview_tab(dataframe):
    total = len(dataframe)
    anomalies = int(dataframe["Ensemble_Anomaly"].sum())
    anomaly_rate = anomalies / total * 100
    high_risk = int((dataframe["RiskLevel"].astype(str) == "High").sum())
    average_score = float(dataframe["RiskScore"].mean())

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Transaksi dianalisis",
        f"{total:,}".replace(",", "."),
    )
    metric_columns[1].metric(
        f"Anomali ensemble · {anomaly_rate:.2f}%".replace(".", ","),
        f"{anomalies:,}".replace(",", "."),
    )
    metric_columns[2].metric(
        "Risiko tinggi",
        f"{high_risk:,}".replace(",", "."),
    )
    metric_columns[3].metric("Rata-rata skor risiko", f"{average_score:.1f} / 100")

    _render_priority_table(dataframe)
    _render_distributions(dataframe)
    _render_concentrations(dataframe)


def _render_priority_table(dataframe):
    section_title(
        "Transaksi prioritas",
        "Sepuluh transaksi dengan skor risiko tertinggi pada cakupan saat ini.",
    )
    priority = dataframe.sort_values(
        ["RiskScore", "AnomalyVoteCount"],
        ascending=False,
    ).head(10)
    columns = safe_columns(priority, PRIORITY_COLUMNS)
    display = priority[columns].copy()
    if "RiskLevel" in display.columns:
        display["RiskLevel"] = display["RiskLevel"].map(translate_risk_level)

    styled_display = display.style.map(
        lambda value: (
            "color: #B4233A; font-weight: 700;"
            if value == "Tinggi"
            else ""
        ),
        subset=["RiskLevel"],
    )

    st.dataframe(
        styled_display,
        width="stretch",
        hide_index=True,
        column_config={
            "TransactionID": st.column_config.TextColumn("ID transaksi"),
            "TransactionAmount": st.column_config.NumberColumn("Nilai transaksi", format="%.2f"),
            "TransactionType": st.column_config.TextColumn("Tipe"),
            "Channel": st.column_config.TextColumn("Kanal"),
            "Location": st.column_config.TextColumn("Lokasi"),
            "AnomalyVoteCount": st.column_config.NumberColumn("Model menandai", format="%d dari 4"),
            "RiskScore": st.column_config.NumberColumn("Skor risiko", format="%.2f"),
            "RiskLevel": st.column_config.TextColumn("Level risiko"),
        },
    )
    st.caption("Buka halaman Antrian tinjauan untuk melihat detail dan faktor utama setiap transaksi.")


def _render_distributions(dataframe):
    section_title(
        "Distribusi risiko",
        "Komposisi level risiko dan sebaran skor untuk transaksi yang sedang ditampilkan.",
    )
    left, right = st.columns(2)

    with left:
        distribution = (
            dataframe["RiskLevel"]
            .astype(str)
            .value_counts()
            .reindex(RISK_ORDER)
            .fillna(0)
            .reset_index()
        )
        distribution.columns = ["RiskLevel", "Count"]
        distribution["Label"] = distribution["RiskLevel"].map(translate_risk_level)
        distribution["Percentage"] = distribution["Count"] / len(dataframe) * 100
        distribution = distribution.sort_values("Count", ascending=True)

        fig_risk = px.bar(
            distribution,
            x="Count",
            y="Label",
            orientation="h",
            color="RiskLevel",
            color_discrete_map=RISK_COLOR_MAP,
            text="Count",
            custom_data=["Percentage"],
            title="Jumlah transaksi per level risiko",
            labels={"Count": "Jumlah transaksi", "Label": "Level risiko"},
        )
        style_plotly_chart(fig_risk, height=330, showlegend=False)
        fig_risk.update_traces(
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{y}<br>%{x:,} transaksi<br>%{customdata[0]:.2f}% dari cakupan<extra></extra>"
            ),
        )
        st.plotly_chart(fig_risk, width="stretch")

    with right:
        fig_score = px.histogram(
            dataframe,
            x="RiskScore",
            nbins=28,
            title="Sebaran skor risiko",
            labels={"RiskScore": "Skor risiko", "count": "Jumlah transaksi"},
        )
        style_plotly_chart(fig_score, height=330, showlegend=False)
        fig_score.update_traces(marker_color=DEFAULT_BAR_COLOR)
        for threshold, label in ((25, "25"), (50, "50"), (75, "75")):
            fig_score.add_vline(
                x=threshold,
                line_width=1,
                line_dash="dash",
                line_color="#7A8491",
                annotation_text=label,
                annotation_position="top",
                annotation_font_size=10,
                annotation_font_color="#5F6B76",
            )
        st.plotly_chart(fig_score, width="stretch")
        st.caption("Garis putus-putus menunjukkan batas skor 25, 50, dan 75.")


def _render_concentrations(dataframe):
    section_title(
        "Konsentrasi risiko",
        "Perbandingan skor antar-kanal dan lokasi dengan kasus risiko tinggi terbanyak.",
    )
    left, right = st.columns(2)

    with left:
        channel = (
            dataframe.groupby("Channel", observed=True)
            .agg(
                AverageRisk=("RiskScore", "mean"),
                Transactions=("TransactionID", "size"),
            )
            .reset_index()
            .sort_values("AverageRisk", ascending=True)
        )
        fig_channel = px.bar(
            channel,
            x="AverageRisk",
            y="Channel",
            orientation="h",
            title="Rata-rata skor risiko per kanal",
            labels={"AverageRisk": "Rata-rata skor", "Channel": "Kanal"},
            text="AverageRisk",
            custom_data=["Transactions"],
        )
        style_plotly_chart(fig_channel, height=310, showlegend=False)
        fig_channel.update_traces(
            marker_color=DEFAULT_BAR_COLOR,
            texttemplate="%{text:.2f}",
            textposition="outside",
            hovertemplate=(
                "%{y}<br>Rata-rata skor: %{x:.2f}<br>Jumlah transaksi: %{customdata[0]:,}<extra></extra>"
            ),
        )
        st.plotly_chart(fig_channel, width="stretch")

    with right:
        high = dataframe[dataframe["RiskLevel"].astype(str) == "High"]
        source = high if not high.empty else dataframe.nlargest(100, "RiskScore")
        title_suffix = "risiko tinggi" if not high.empty else "100 skor tertinggi"
        locations = (
            source.groupby("Location", observed=True)
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
            .head(8)
            .sort_values("Count", ascending=True)
        )
        fig_location = px.bar(
            locations,
            x="Count",
            y="Location",
            orientation="h",
            title=f"Lokasi dengan {title_suffix}",
            labels={"Count": "Jumlah transaksi", "Location": "Lokasi"},
            text="Count",
        )
        style_plotly_chart(fig_location, height=310, showlegend=False)
        fig_location.update_traces(
            marker_color=RISK_COLOR_MAP["High"],
            textposition="outside",
            cliponaxis=False,
        )
        st.plotly_chart(fig_location, width="stretch")
