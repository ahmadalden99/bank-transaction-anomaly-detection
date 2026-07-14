from datetime import date, datetime
from ipaddress import ip_address

import pandas as pd
import streamlit as st

from dashboard.components.cards import metric_card, risk_badge_html, risk_metric_card
from dashboard.components.layout import info_box, section_title, subsection_title
from src.prediction import predict_new_transaction
from src.utils import humanize_feature_name, safe_columns, stringify_value_column


GENERATED_FEATURE_COLUMNS = [
    "Amount_to_Balance_Ratio",
    "Amount_to_TypeChannelBalanceGroupAvg_Ratio",
    "IsLongDuration",
    "IsHighLoginAttempts",
    "BalanceGroup_ByType",
    "AccountTxCount",
    "DeviceTxCount",
    "MerchantTxCount",
    "IPTxCount",
]

MODEL_NAMES = {
    "IF_Anomaly": "Isolation Forest",
    "LOF_Anomaly": "Local Outlier Factor",
    "OCSVM_Anomaly": "One-Class SVM",
    "MCD_Anomaly": "MCD Envelope",
}


def render_prediction_tab(dataframe, load_prediction_bundle):
    info_box(
        """
        <strong>Prediksi pola, bukan vonis fraud.</strong> Sistem membandingkan transaksi baru dengan pola
        pada data pelatihan. Hasilnya adalah sinyal prioritas untuk tinjauan manusia, bukan probabilitas fraud.
        """,
        tone="info",
    )

    _initialize_prediction_state(dataframe)
    _render_preset_controls(dataframe)
    input_data, submitted = _render_prediction_form(dataframe)

    if submitted:
        errors, warnings = _validate_prediction_input(input_data, dataframe)
        if errors:
            for error in errors:
                st.error(error)
        else:
            for warning in warnings:
                st.warning(warning)

            try:
                with st.spinner("Empat model sedang menilai pola transaksi…"):
                    prediction_bundle, missing_files = load_prediction_bundle()
                    if prediction_bundle is None:
                        raise FileNotFoundError(
                            "Artifact model belum lengkap: " + ", ".join(missing_files)
                        )
                    new_raw, prediction_result = predict_new_transaction(
                        input_data,
                        prediction_bundle,
                    )
            except Exception as error:
                st.error(
                    "Prediksi belum dapat dijalankan. Periksa artifact model dan nilai input. "
                    f"Detail: {error}"
                )
            else:
                st.session_state["prediction_payload"] = {
                    "raw": new_raw,
                    "result": prediction_result,
                    "input": input_data,
                }

    prediction_payload = st.session_state.get("prediction_payload")
    if prediction_payload:
        _render_prediction_result(
            prediction_payload["raw"],
            prediction_payload["result"],
            prediction_payload["input"],
            dataframe,
        )


def _initialize_prediction_state(dataframe):
    defaults = _prediction_defaults(dataframe, suspicious=False)
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _prediction_defaults(dataframe, suspicious=False):
    if suspicious:
        amount = float(dataframe["TransactionAmount"].quantile(0.99))
        balance = float(dataframe["AccountBalance"].quantile(0.10))
        duration = int(dataframe["TransactionDuration"].quantile(0.95))
        login_attempts = int(dataframe["LoginAttempts"].max())
    else:
        amount = float(dataframe["TransactionAmount"].median())
        balance = float(dataframe["AccountBalance"].median())
        duration = int(dataframe["TransactionDuration"].median())
        login_attempts = int(dataframe["LoginAttempts"].median())

    def first_option(column):
        options = sorted(dataframe[column].dropna().astype(str).unique().tolist())
        return options[0] if options else ""

    return {
        "pred_amount": round(amount, 2),
        "pred_balance": round(max(balance, 0.01), 2),
        "pred_duration": max(duration, 1),
        "pred_logins": max(login_attempts, 1),
        "pred_age": int(dataframe["CustomerAge"].median()),
        "pred_date": date.today(),
        "pred_time": datetime.now().time().replace(second=0, microsecond=0),
        "pred_type": first_option("TransactionType"),
        "pred_channel": first_option("Channel"),
        "pred_location": first_option("Location"),
        "pred_occupation": first_option("CustomerOccupation"),
        "pred_account": "NEW_ACCOUNT",
        "pred_device": "NEW_DEVICE",
        "pred_merchant": "NEW_MERCHANT",
        "pred_ip": "0.0.0.0",
    }


def _apply_prediction_preset(dataframe, suspicious):
    st.session_state.update(_prediction_defaults(dataframe, suspicious=suspicious))
    st.session_state.pop("prediction_payload", None)


def _clear_prediction_result():
    st.session_state.pop("prediction_payload", None)


def _render_preset_controls(dataframe):
    section_title(
        "Bangun skenario",
        "Mulai dari contoh referensi atau isi setiap nilai secara manual.",
    )
    preset_left, preset_middle, preset_right = st.columns([1, 1, 2.2])
    with preset_left:
        st.button(
            "Isi contoh umum",
            on_click=_apply_prediction_preset,
            args=(dataframe, False),
            width="stretch",
            type="secondary",
        )
    with preset_middle:
        st.button(
            "Isi contoh tidak biasa",
            on_click=_apply_prediction_preset,
            args=(dataframe, True),
            width="stretch",
            type="secondary",
        )
    with preset_right:
        st.caption(
            "Preset hanya membantu mengisi form. Model tetap menghitung hasil dari semua nilai yang terlihat."
        )


def _render_prediction_form(dataframe):
    with st.form("prediction_form", border=True):
        col_left, col_right = st.columns(2)

        with col_left:
            subsection_title("Nilai & perilaku")
            st.number_input(
                "Nilai transaksi",
                min_value=0.0,
                step=10.0,
                key="pred_amount",
                help=_range_help(dataframe["TransactionAmount"]),
            )
            st.number_input(
                "Saldo rekening",
                min_value=0.0,
                step=100.0,
                key="pred_balance",
                help=_range_help(dataframe["AccountBalance"]),
            )

            behavior_left, behavior_right = st.columns(2)
            with behavior_left:
                st.number_input(
                    "Durasi (detik)",
                    min_value=1,
                    step=1,
                    key="pred_duration",
                    help=_range_help(dataframe["TransactionDuration"], decimals=0),
                )
                st.number_input(
                    "Usia nasabah",
                    min_value=1,
                    max_value=100,
                    step=1,
                    key="pred_age",
                    help=_range_help(dataframe["CustomerAge"], decimals=0),
                )
            with behavior_right:
                st.number_input(
                    "Percobaan login",
                    min_value=1,
                    step=1,
                    key="pred_logins",
                    help=_range_help(dataframe["LoginAttempts"], decimals=0),
                )

            date_left, date_right = st.columns(2)
            with date_left:
                st.date_input("Tanggal", key="pred_date")
            with date_right:
                st.time_input("Waktu", key="pred_time")

        with col_right:
            subsection_title("Konteks transaksi")
            _select_from_dataframe(dataframe, "TransactionType", "Tipe transaksi", "pred_type")
            _select_from_dataframe(dataframe, "Channel", "Kanal", "pred_channel")
            _select_from_dataframe(dataframe, "Location", "Lokasi", "pred_location")
            _select_from_dataframe(
                dataframe,
                "CustomerOccupation",
                "Pekerjaan nasabah",
                "pred_occupation",
            )

        with st.expander("Identitas teknis (opsional)"):
            id_col1, id_col2 = st.columns(2)
            with id_col1:
                st.text_input("Account ID (opsional)", key="pred_account")
                st.text_input("Device ID (opsional)", key="pred_device")
            with id_col2:
                st.text_input("Merchant ID (opsional)", key="pred_merchant")
                st.text_input("IP Address (opsional)", key="pred_ip")
            st.caption(
                "ID digunakan untuk fitur frekuensi. ID yang belum dikenal diperlakukan sebagai entitas baru."
            )

        submitted = st.form_submit_button(
            "Analisis risiko transaksi",
            width="stretch",
            type="primary",
        )

    input_data = {
        "TransactionAmount": st.session_state["pred_amount"],
        "AccountBalance": st.session_state["pred_balance"],
        "TransactionDuration": st.session_state["pred_duration"],
        "LoginAttempts": st.session_state["pred_logins"],
        "CustomerAge": st.session_state["pred_age"],
        "TransactionDate": f"{st.session_state['pred_date']} {st.session_state['pred_time']}",
        "TransactionType": st.session_state["pred_type"],
        "Channel": st.session_state["pred_channel"],
        "Location": st.session_state["pred_location"],
        "CustomerOccupation": st.session_state["pred_occupation"],
        "AccountID": st.session_state["pred_account"].strip() or "NEW_ACCOUNT",
        "DeviceID": st.session_state["pred_device"].strip() or "NEW_DEVICE",
        "MerchantID": st.session_state["pred_merchant"].strip() or "NEW_MERCHANT",
        "IP Address": st.session_state["pred_ip"].strip() or "0.0.0.0",
    }
    return input_data, submitted


def _select_from_dataframe(dataframe, column, label, key):
    options = sorted(dataframe[column].dropna().astype(str).unique().tolist())
    if st.session_state.get(key) not in options and options:
        st.session_state[key] = options[0]
    st.selectbox(label, options=options, key=key)


def _range_help(series, decimals=2):
    return (
        "Rentang data pelatihan: "
        f"{series.min():,.{decimals}f} – {series.max():,.{decimals}f}"
    )


def _validate_prediction_input(input_data, dataframe):
    errors = []
    warnings = []

    if input_data["TransactionAmount"] <= 0:
        errors.append("Nilai transaksi harus lebih besar dari 0.")
    if input_data["AccountBalance"] <= 0:
        errors.append("Saldo rekening harus lebih besar dari 0 agar rasio transaksi valid.")

    try:
        ip_address(input_data["IP Address"])
    except ValueError:
        errors.append("IP Address tidak valid. Gunakan format seperti 192.168.1.10.")

    range_checks = (
        ("TransactionAmount", "Nilai transaksi"),
        ("AccountBalance", "Saldo rekening"),
        ("TransactionDuration", "Durasi"),
        ("LoginAttempts", "Percobaan login"),
        ("CustomerAge", "Usia nasabah"),
    )
    for column, label in range_checks:
        value = input_data[column]
        if value < dataframe[column].min() or value > dataframe[column].max():
            warnings.append(
                f"{label} berada di luar rentang data pelatihan; interpretasikan hasil dengan lebih hati-hati."
            )

    return errors, warnings


def _render_prediction_result(new_raw, prediction_result, input_data, dataframe):
    st.write("")
    section_title(
        "Hasil analisis",
        "Keputusan ensemble, perbandingan terhadap median data, dan tindakan yang disarankan.",
    )

    _render_decision_banner(prediction_result)
    _render_result_metrics(prediction_result)
    _render_recommended_action(prediction_result)

    _render_prediction_evidence(prediction_result, input_data, dataframe)

    with st.expander("Lihat detail teknis & fitur hasil rekayasa"):
        _render_technical_details(new_raw, prediction_result)

    st.button(
        "Bersihkan hasil analisis",
        on_click=_clear_prediction_result,
        width="stretch",
        type="secondary",
    )


def _render_decision_banner(prediction_result):
    is_anomaly = prediction_result["Ensemble_Anomaly"] == 1
    title = "Perlu pemeriksaan lebih lanjut" if is_anomaly else "Tidak ada sinyal ensemble kuat"
    description = (
        "Minimal dua model mendeteksi pola tidak biasa pada transaksi ini."
        if is_anomaly
        else "Kurang dari dua model menandai transaksi sebagai anomali."
    )
    banner_class = "decision-alert" if is_anomaly else "decision-clear"

    st.markdown(
        f"""
        <div class="decision-banner {banner_class}">
            <div>
                <div class="decision-kicker">Hasil ensemble</div>
                <div class="decision-title">{title}</div>
                <div class="decision-copy">{description}</div>
            </div>
            <div>{risk_badge_html(prediction_result['RiskLevel'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_result_metrics(prediction_result):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card(
            "Skor risiko",
            f"{prediction_result['RiskScore']:.2f} / 100",
            "Skor gabungan model",
        )
    with col2:
        risk_metric_card("Level risiko", prediction_result["RiskLevel"])
    with col3:
        metric_card(
            "Model menandai",
            f"{prediction_result['AnomalyVoteCount']} dari 4",
            "Keputusan individual",
        )
    with col4:
        status = "Tinjau" if prediction_result["Ensemble_Anomaly"] else "Pantau"
        metric_card("Tindakan awal", status, "Bukan konfirmasi fraud")


def _render_recommended_action(prediction_result):
    risk_level = prediction_result["RiskLevel"]
    if risk_level == "High":
        message = (
            "<strong>Tindakan:</strong> tahan untuk verifikasi manual, konfirmasi identitas, dan tinjau perangkat serta histori rekening."
        )
        tone = "danger"
    elif prediction_result["Ensemble_Anomaly"] == 1:
        message = (
            "<strong>Tindakan:</strong> masukkan ke antrian tinjauan dan cari transaksi serupa dalam histori rekening."
        )
        tone = "warning"
    elif risk_level == "Medium":
        message = "<strong>Tindakan:</strong> pantau dan lakukan verifikasi tambahan bila konteks lain mencurigakan."
        tone = "warning"
    else:
        message = "<strong>Tindakan:</strong> lanjutkan pemantauan normal; belum ada kebutuhan eskalasi otomatis."
        tone = "neutral"

    info_box(message, tone=tone)


def _render_prediction_evidence(prediction_result, input_data, dataframe):
    section_title(
        "Bukti pendukung",
        "Kesepakatan model dan posisi nilai input terhadap median dataset.",
    )
    evidence_left, evidence_right = st.columns([1, 1.15])

    with evidence_left:
        decision_table = pd.DataFrame(
            {
                "Model": list(MODEL_NAMES.values()),
                "Keputusan": [
                    "Ditandai" if prediction_result[column] else "Tidak ditandai"
                    for column in MODEL_NAMES
                ],
            }
        )
        subsection_title("Keputusan empat model")
        st.dataframe(
            decision_table,
            width="stretch",
            height=260,
            hide_index=True,
            column_config={
                "Model": st.column_config.TextColumn("Model", width="large"),
                "Keputusan": st.column_config.TextColumn("Keputusan", width="medium"),
            },
        )

    with evidence_right:
        compare_columns = [
            "TransactionAmount",
            "AccountBalance",
            "TransactionDuration",
            "LoginAttempts",
            "CustomerAge",
        ]
        comparison = []
        for column in compare_columns:
            current_value = float(input_data[column])
            median_value = float(dataframe[column].median())
            delta = (
                (current_value - median_value) / median_value * 100
                if median_value != 0
                else 0
            )
            comparison.append(
                {
                    "Atribut": humanize_feature_name(column),
                    "Input": current_value,
                    "Median dataset": median_value,
                    "Selisih": delta,
                }
            )

        st.dataframe(
            pd.DataFrame(comparison),
            width="stretch",
            height=260,
            hide_index=True,
            column_config={
                "Atribut": st.column_config.TextColumn("Atribut", width="large"),
                "Input": st.column_config.NumberColumn("Input", format="%.2f"),
                "Median dataset": st.column_config.NumberColumn("Median", format="%.2f"),
                "Selisih": st.column_config.NumberColumn("vs median", format="%+.1f%%"),
            },
        )


def _render_technical_details(new_raw, prediction_result):
    generated_columns = safe_columns(new_raw, GENERATED_FEATURE_COLUMNS)
    generated_table = new_raw[generated_columns].T.reset_index()
    generated_table.columns = ["Fitur", "Nilai"]
    generated_table["Fitur"] = generated_table["Fitur"].map(humanize_feature_name)
    generated_table = stringify_value_column(generated_table, "Nilai")

    score_table = pd.DataFrame(
        {
            "Metrik": [
                "IF Score",
                "LOF Score",
                "OCSVM Score",
                "MCD Score",
                "Average normalized score",
                "Vote score",
            ],
            "Nilai": [
                prediction_result["IF_Score"],
                prediction_result["LOF_Score"],
                prediction_result["OCSVM_Score"],
                prediction_result["MCD_Score"],
                prediction_result["Average_AnomalyScore_Norm"],
                prediction_result["VoteScore"],
            ],
        }
    )

    technical_left, technical_right = st.columns(2)
    with technical_left:
        st.dataframe(generated_table, width="stretch", hide_index=True)
    with technical_right:
        st.dataframe(
            score_table,
            width="stretch",
            hide_index=True,
            column_config={
                "Nilai": st.column_config.NumberColumn("Nilai", format="%.5f")
            },
        )
