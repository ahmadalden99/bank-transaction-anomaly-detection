# Analisis Anomali Transaksi Bank

Proyek portofolio data science untuk mendeteksi pola transaksi tidak biasa menggunakan ensemble empat model unsupervised. Dashboard membantu pengguna melihat transaksi prioritas, memahami hasil model, menjelajahi data, dan menguji transaksi baru.

> Hasil model merupakan sinyal anomaly detection dan bukan label fraud aktual.

## Halaman dashboard

- **Ringkasan** — metrik utama, transaksi prioritas, distribusi skor, serta konsentrasi risiko.
- **Antrian tinjauan** — pemilihan transaksi, profil kasus, keputusan setiap model, dan faktor SHAP.
- **Penjelasan model** — tingkat kepentingan fitur global dan pola antar-level risiko.
- **Data transaksi** — pencarian, pilihan kolom, dan unduhan CSV.
- **Uji transaksi** — form transaksi baru, validasi input, perbandingan dengan median, dan hasil ensemble.

## Menjalankan aplikasi

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Kemudian buka `http://localhost:8501`.

## Struktur proyek

```text
.
├── app.py
├── dashboard/
│   ├── app.py
│   ├── components/
│   ├── styles/
│   └── tabs/
├── src/
├── data/raw/
├── notebooks/
├── artifacts/
├── outputs/
└── requirements.txt
```

Aturan visual dan pertimbangan UX dicatat di `DESIGN.md`.
