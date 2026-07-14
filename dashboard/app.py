from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


st.set_page_config(
    page_title="Analisis Anomali Transaksi Bank",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


from dashboard.components.filters import (
    render_prediction_sidebar,
    render_sidebar_filters,
)
from dashboard.components.layout import (
    load_css,
    render_empty_state,
    render_page_header,
    render_sidebar_brand,
)
from dashboard.tabs.data_explorer import render_data_explorer_tab
from dashboard.tabs.explainability import render_explainability_tab
from dashboard.tabs.overview import render_overview_tab
from dashboard.tabs.prediction import render_prediction_tab
from dashboard.tabs.transactions import render_transactions_tab
from src.data_loader import load_feature_importance, load_transaction_results
from src.prediction import load_prediction_artifacts


CSS_PATH = Path(__file__).resolve().parent / "styles" / "main.css"
PAGES = [
    "Ringkasan",
    "Antrian tinjauan",
    "Penjelasan model",
    "Data transaksi",
    "Uji transaksi",
]


@st.cache_data
def get_transaction_results():
    return load_transaction_results()


@st.cache_data
def get_feature_importance():
    return load_feature_importance()


@st.cache_resource
def get_prediction_bundle():
    return load_prediction_artifacts()


def _render_navigation():
    st.sidebar.markdown(
        '<div class="sidebar-section-label navigation-label">Navigasi</div>',
        unsafe_allow_html=True,
    )
    return st.sidebar.radio(
        "Navigasi",
        options=PAGES,
        label_visibility="collapsed",
        key="primary_navigation",
    )


def _render_selected_page(page_name, dataframe, filtered_dataframe, feature_importance):
    if page_name != "Uji transaksi" and filtered_dataframe.empty:
        render_empty_state()
        return

    if page_name == "Ringkasan":
        render_overview_tab(filtered_dataframe)
    elif page_name == "Antrian tinjauan":
        render_transactions_tab(filtered_dataframe)
    elif page_name == "Penjelasan model":
        render_explainability_tab(filtered_dataframe, feature_importance)
    elif page_name == "Data transaksi":
        render_data_explorer_tab(filtered_dataframe)
    elif page_name == "Uji transaksi":
        render_prediction_tab(dataframe, get_prediction_bundle)


def main():
    load_css(CSS_PATH)

    try:
        dataframe = get_transaction_results()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    feature_importance = get_feature_importance()

    render_sidebar_brand()
    selected_page = _render_navigation()

    if selected_page == "Uji transaksi":
        filtered_dataframe = dataframe
        render_prediction_sidebar()
    else:
        filtered_dataframe = render_sidebar_filters(dataframe)

    filtered_anomalies = (
        int(filtered_dataframe["Ensemble_Anomaly"].sum())
        if "Ensemble_Anomaly" in filtered_dataframe.columns
        else 0
    )
    filtered_high_risk = (
        int((filtered_dataframe["RiskLevel"].astype(str) == "High").sum())
        if "RiskLevel" in filtered_dataframe.columns
        else 0
    )

    render_page_header(
        selected_page,
        total_transactions=len(dataframe),
        filtered_transactions=len(filtered_dataframe),
        filtered_anomalies=filtered_anomalies,
        filtered_high_risk=filtered_high_risk,
    )

    _render_selected_page(
        selected_page,
        dataframe,
        filtered_dataframe,
        feature_importance,
    )

    st.caption(
        "Catatan: skor anomali membantu menentukan prioritas tinjauan dan bukan label fraud terkonfirmasi."
    )


if __name__ == "__main__":
    main()
