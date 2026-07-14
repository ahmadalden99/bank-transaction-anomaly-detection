from math import ceil, floor

import streamlit as st

from src.config import RISK_ORDER
from src.utils import translate_risk_level


FILTER_DEFAULTS = {
    "quick_scope": "Semua transaksi",
    "filter_risk": [],
    "filter_type": [],
    "filter_channel": [],
    "filter_occupation": [],
    "filter_location": [],
}


def _reset_filters(score_bounds):
    for key, value in FILTER_DEFAULTS.items():
        st.session_state[key] = value.copy() if isinstance(value, list) else value
    st.session_state["filter_score"] = score_bounds


def _keep_valid_values(key, options):
    current = st.session_state.get(key, [])
    valid = [value for value in current if value in options]
    if current != valid:
        st.session_state[key] = valid


def render_sidebar_filters(dataframe):
    minimum = float(dataframe["RiskScore"].min())
    maximum = float(dataframe["RiskScore"].max())
    score_bounds = (
        float(floor(minimum * 10) / 10),
        float(ceil(maximum * 10) / 10),
    )

    for key, value in FILTER_DEFAULTS.items():
        st.session_state.setdefault(
            key,
            value.copy() if isinstance(value, list) else value,
        )
    st.session_state.setdefault("filter_score", score_bounds)

    st.sidebar.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="sidebar-section-label">Cakupan data</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.selectbox(
        "Tampilkan",
        options=["Semua transaksi", "Anomali ensemble", "Risiko tinggi"],
        key="quick_scope",
    )

    risk_options = [
        risk
        for risk in RISK_ORDER
        if risk in dataframe["RiskLevel"].astype("string").dropna().unique().tolist()
    ]
    type_options = sorted(dataframe["TransactionType"].dropna().astype(str).unique())
    channel_options = sorted(dataframe["Channel"].dropna().astype(str).unique())
    occupation_options = sorted(dataframe["CustomerOccupation"].dropna().astype(str).unique())
    location_options = sorted(dataframe["Location"].dropna().astype(str).unique())

    for key, options in (
        ("filter_risk", risk_options),
        ("filter_type", type_options),
        ("filter_channel", channel_options),
        ("filter_occupation", occupation_options),
        ("filter_location", location_options),
    ):
        _keep_valid_values(key, options)

    st.sidebar.markdown(
        '<div class="sidebar-section-label filter-heading">Filter</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.multiselect(
        "Level risiko",
        options=risk_options,
        format_func=translate_risk_level,
        key="filter_risk",
        placeholder="Semua level",
    )
    st.sidebar.multiselect(
        "Tipe transaksi",
        options=type_options,
        key="filter_type",
        placeholder="Semua tipe",
    )
    st.sidebar.multiselect(
        "Kanal",
        options=channel_options,
        key="filter_channel",
        placeholder="Semua kanal",
    )

    with st.sidebar.expander("Profil dan lokasi"):
        st.multiselect(
            "Pekerjaan nasabah",
            options=occupation_options,
            key="filter_occupation",
            placeholder="Semua pekerjaan",
        )
        st.multiselect(
            "Lokasi",
            options=location_options,
            key="filter_location",
            placeholder="Semua lokasi",
        )

    if minimum < maximum:
        current_score = st.session_state.get("filter_score", score_bounds)
        current_score = (
            max(score_bounds[0], min(float(current_score[0]), score_bounds[1])),
            max(score_bounds[0], min(float(current_score[1]), score_bounds[1])),
        )
        if current_score[0] > current_score[1]:
            current_score = score_bounds
        st.session_state["filter_score"] = current_score
        st.sidebar.slider(
            "Skor risiko",
            min_value=score_bounds[0],
            max_value=score_bounds[1],
            step=0.5,
            key="filter_score",
        )
    else:
        st.session_state["filter_score"] = score_bounds

    mask = dataframe["RiskScore"].between(*st.session_state["filter_score"])
    for state_key, column in (
        ("filter_risk", "RiskLevel"),
        ("filter_type", "TransactionType"),
        ("filter_channel", "Channel"),
        ("filter_occupation", "CustomerOccupation"),
        ("filter_location", "Location"),
    ):
        selected = st.session_state[state_key]
        if selected:
            mask &= dataframe[column].astype(str).isin(selected)

    if st.session_state["quick_scope"] == "Anomali ensemble":
        mask &= dataframe["Ensemble_Anomaly"].eq(1)
    elif st.session_state["quick_scope"] == "Risiko tinggi":
        mask &= dataframe["RiskLevel"].astype(str).eq("High")

    filtered = dataframe.loc[mask].copy()

    st.sidebar.markdown(
        f'<p class="filter-count"><strong>{len(filtered):,}</strong> transaksi ditampilkan</p>',
        unsafe_allow_html=True,
    )
    st.sidebar.button(
        "Reset filter",
        on_click=_reset_filters,
        args=(score_bounds,),
        width="stretch",
    )
    return filtered


def render_prediction_sidebar():
    st.sidebar.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="sidebar-section-label">Tentang pengujian</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption(
        "Transaksi baru dibandingkan dengan pola pada seluruh dataset pelatihan. Filter halaman analisis tidak digunakan."
    )
