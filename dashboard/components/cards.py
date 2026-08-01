from html import escape

import streamlit as st


RISK_BADGE_CLASSES = {
    "High": "risk-high",
    "Medium": "risk-medium",
    "Low-Medium": "risk-lowmedium",
    "Low": "risk-low",
}

RISK_LABELS = {
    "High": "Tinggi",
    "Medium": "Sedang",
    "Low-Medium": "Rendah–sedang",
    "Low": "Rendah",
}


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{escape(str(label))}</div>
            <div class="metric-value">{escape(str(value))}</div>
            <div class="metric-note">{escape(str(note))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge_html(risk_level, translated=True):
    risk_text = str(risk_level)
    css_class = RISK_BADGE_CLASSES.get(risk_text, "risk-unknown")
    display_text = RISK_LABELS.get(risk_text, "Tidak diketahui") if translated else risk_text
    return f'<span class="risk-badge {css_class}">{escape(display_text)}</span>'


def risk_metric_card(label, risk_level, note="Kategori risiko"):
    st.markdown(
        f"""
        <div class="metric-card metric-risk">
            <div class="metric-label">{escape(str(label))}</div>
            <div class="risk-card-value">{risk_badge_html(risk_level)}</div>
            <div class="metric-note">{escape(str(note))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
