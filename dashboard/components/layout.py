from html import escape

import streamlit as st


PAGE_META = {
    "Ringkasan": (
        "Ringkasan hasil deteksi anomali",
        "Ikhtisar transaksi yang memerlukan perhatian dan pola risiko pada data.",
    ),
    "Antrian tinjauan": (
        "Antrian tinjauan transaksi",
        "Pilih transaksi untuk memeriksa profil, keputusan model, dan faktor pendorongnya.",
    ),
    "Penjelasan model": (
        "Penjelasan hasil model",
        "Lihat fitur yang paling berpengaruh dan perbedaan pola antar-level risiko.",
    ),
    "Data transaksi": (
        "Data hasil analisis",
        "Cari, periksa, pilih kolom, dan unduh data setelah proses deteksi anomali.",
    ),
    "Uji transaksi": (
        "Uji transaksi baru",
        "Bandingkan transaksi baru dengan pola yang dipelajari oleh empat model anomali.",
    ),
}


def load_css(css_path):
    if not css_path.exists():
        st.warning(f"CSS tidak ditemukan: {css_path}")
        return
    st.markdown(
        f"<style>{css_path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


def render_sidebar_brand():
    st.sidebar.markdown(
        """
        <div class="project-identity">
            <div class="project-name">Analisis Anomali<br>Transaksi Bank</div>
            <div class="project-type">Proyek portofolio data science</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(
    page_name,
    total_transactions,
    filtered_transactions,
    filtered_anomalies,
    filtered_high_risk,
):
    title, description = PAGE_META.get(page_name, PAGE_META["Ringkasan"])
    del filtered_anomalies, filtered_high_risk
    scope_text = (
        f"{total_transactions:,} transaksi"
        if filtered_transactions == total_transactions
        else f"{filtered_transactions:,} dari {total_transactions:,} transaksi"
    )
    st.markdown(
        f"""
        <header class="page-header">
            <h1>{escape(title)}</h1>
            <p>{escape(description)}</p>
            <div class="page-context">
                <span>Cakupan: <strong>{escape(scope_text)}</strong></span>
                <span>Metode: <strong>ensemble empat model unsupervised</strong></span>
            </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def section_title(title, subtitle=None):
    subtitle_html = (
        f'<p class="section-subtitle">{escape(str(subtitle))}</p>'
        if subtitle
        else ""
    )
    st.markdown(
        f"""
        <div class="section-heading">
            <h2>{escape(str(title))}</h2>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_subtitle(text):
    st.markdown(
        f'<p class="section-subtitle standalone-subtitle">{escape(str(text))}</p>',
        unsafe_allow_html=True,
    )


def subsection_title(title):
    st.markdown(
        f'<h3 class="subsection-title">{escape(str(title))}</h3>',
        unsafe_allow_html=True,
    )


def info_box(markdown_text, tone="info"):
    st.markdown(
        f'<div class="info-box info-{escape(str(tone))}">{markdown_text}</div>',
        unsafe_allow_html=True,
    )


def render_empty_state(
    title="Tidak ada transaksi yang cocok",
    description="Ubah atau reset filter untuk menampilkan data kembali.",
):
    st.markdown(
        f"""
        <div class="empty-state">
            <h2>{escape(str(title))}</h2>
            <p>{escape(str(description))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
