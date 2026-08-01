# Bank Transaction Anomaly Detection

Proyek ini dibuat untuk membantu menyaring transaksi bank yang memiliki pola tidak biasa. Empat model *unsupervised learning* digabungkan untuk memberi skor risiko, lalu hasilnya disajikan melalui dashboard Streamlit agar transaksi yang perlu diperiksa lebih lanjut lebih mudah ditemukan.

> **Catatan:** hasil model adalah sinyal anomali, bukan bukti bahwa sebuah transaksi merupakan fraud.

## Yang bisa dilakukan di dashboard

- Melihat jumlah transaksi, distribusi tingkat risiko, dan ringkasan hasil deteksi.
- Menyaring transaksi berdasarkan level risiko, tipe transaksi, channel, dan lokasi.
- Membuka detail transaksi beserta keputusan dari masing-masing model.
- Melihat fitur yang paling berpengaruh melalui model surrogate XGBoost.
- Menjelajahi data dan mengunduh hasil yang sudah difilter.
- Menguji transaksi baru dan membandingkannya dengan pola pada data yang tersedia.

## Data yang digunakan

Dataset mentah berisi **2.512 transaksi dan 16 kolom**, mulai dari nominal dan waktu transaksi hingga informasi akun, perangkat, merchant, channel, lokasi, durasi transaksi, dan jumlah percobaan login. Pemeriksaan awal tidak menemukan missing value maupun baris duplikat.

`TransactionID` tetap dipertahankan sebagai penanda transaksi, tetapi tidak digunakan sebagai input model. Hal yang sama berlaku untuk ID mentah lain karena angka atau kode pada ID tidak menunjukkan hubungan numerik yang bermakna.

## Alur pemrosesan data

Seluruh proses eksplorasi, feature engineering, dan pelatihan model berada di `notebooks/coba_model.ipynb`. Secara ringkas, alurnya seperti ini:

### 1. Feature engineering

- **Waktu transaksi:** `TransactionDate` diubah menjadi jam, hari, bulan, dan penanda akhir pekan. Jam, hari, dan bulan kemudian diubah dengan sine-cosine encoding agar pola siklik tetap terbaca oleh model. Contohnya, pukul 23.00 tetap dianggap dekat dengan pukul 00.00.
- **Konteks nominal:** dibuat rasio nominal transaksi terhadap saldo akun. Nominal juga dibandingkan dengan rata-rata transaksi pada kombinasi tipe transaksi, channel, dan kelompok saldo yang sama.
- **Perilaku login:** transaksi dengan tiga atau lebih percobaan login diberi penanda `IsHighLoginAttempts`.
- **Durasi transaksi:** `IsLongDuration` menandai transaksi dengan durasi di atas kuartil ketiga dataset.
- **Frekuensi penggunaan:** ID akun, perangkat, merchant, dan alamat IP diubah menjadi jumlah kemunculannya. Dengan cara ini, model menggunakan pola frekuensi dan bukan kode ID mentah.

Setelah seleksi fitur, input model terdiri dari **20 fitur numerik** dan **5 fitur kategorikal**. Fitur kategorikalnya adalah tipe transaksi, channel, lokasi, pekerjaan nasabah, dan kelompok saldo berdasarkan tipe transaksi.

### 2. Preprocessing

Fitur numerik distandardisasi menggunakan `StandardScaler`, sedangkan fitur kategorikal diubah menjadi kolom numerik menggunakan one-hot encoding. Seluruh hasilnya digabungkan menjadi satu matriks input. Fitur yang tidak memiliki variasi juga dibuang karena tidak memberi informasi untuk membedakan pola transaksi.

Objek preprocessing, urutan kolom, threshold, dan frequency map disimpan di folder `artifacts/`. Dashboard menggunakan objek yang sama saat menerima transaksi baru, sehingga proses transformasinya konsisten dengan data saat model dilatih.

### 3. Deteksi anomali

Karena dataset tidak memiliki label fraud, pendekatan yang digunakan adalah *unsupervised anomaly detection*. Empat algoritma melihat pola tidak biasa dari sudut yang berbeda:

1. **Isolation Forest** mencari observasi yang lebih mudah diisolasi dari mayoritas data.
2. **Local Outlier Factor** membandingkan kepadatan suatu transaksi dengan transaksi di sekitarnya.
3. **One-Class SVM** mempelajari batas pola transaksi yang dianggap normal.
4. **MCD / Elliptic Envelope** mencari penyimpangan multivariat menggunakan estimasi kovarians yang robust.

Eksperimen menggunakan contamination rate sebesar **5%** pada masing-masing model. Nilai ini merupakan asumsi pemodelan untuk membantu model menentukan batas anomali, bukan pernyataan bahwa 5% transaksi pasti fraud.

### 4. Ensemble dan skor risiko

Setiap model memberi satu suara: `1` untuk anomali dan `0` untuk normal. Transaksi ditandai sebagai `Ensemble_Anomaly` ketika sedikitnya dua dari empat model memberikan sinyal anomali.

Skor mentah dari setiap model memiliki skala yang berbeda, sehingga semuanya dinormalisasi ke rentang 0-1 dengan `MinMaxScaler`. Skor risiko akhir menggabungkan kesepakatan model dan kekuatan skor anomalinya:

```text
VoteScore = AnomalyVoteCount / 4
RiskScore = (0.7 * VoteScore + 0.3 * AverageNormalizedAnomalyScore) * 100
```

Hasilnya dibagi menjadi empat tingkat prioritas:

| RiskScore | Risk level |
| ---: | --- |
| `< 25` | Low |
| `25 - < 50` | Low-Medium |
| `50 - < 75` | Medium |
| `>= 75` | High |

Pada hasil yang tersimpan saat ini, ensemble menandai **128 dari 2.512 transaksi** sebagai anomali. Distribusi level risikonya adalah 2.260 Low, 207 Low-Medium, 27 Medium, dan 18 High.

### 5. Explainability

XGBoost digunakan sebagai model surrogate untuk mempelajari keputusan akhir ensemble, bukan untuk menentukan label fraud. Feature importance dipakai untuk melihat pola secara global, sedangkan SHAP menunjukkan fitur yang paling mendorong setiap transaksi ke arah anomali atau normal. Karena target surrogate berasal dari output ensemble, evaluasinya menunjukkan seberapa baik XGBoost meniru ensemble dan bukan akurasi deteksi fraud di dunia nyata.

## Struktur proyek

```text
.
|-- app.py                 # Entry point aplikasi
|-- dashboard/             # Halaman, komponen, dan styling Streamlit
|-- src/                   # Konfigurasi, pemuatan data, dan proses prediksi
|-- data/raw/              # Dataset transaksi asli
|-- notebooks/             # Eksplorasi data dan pengembangan model
|-- artifacts/             # Model serta objek preprocessing tersimpan
|-- outputs/               # Hasil deteksi dan feature importance
|-- DESIGN.md              # Catatan keputusan visual dan UX
`-- requirements.txt       # Daftar dependency Python
```

## Model dan output

Model yang sudah dilatih tersimpan di folder `artifacts/`, sehingga dashboard dapat langsung digunakan setelah seluruh dependency terpasang. Hasil akhir deteksi berada di `outputs/bank_transaction_anomaly_final_result.csv`, sedangkan proses eksperimen dan pemodelan dapat dilihat di `notebooks/coba_model.ipynb`.

Proyek ini berfokus pada proses *screening* dan prioritas tinjauan. Untuk penggunaan nyata, hasilnya tetap perlu divalidasi menggunakan label fraud, masukan analis, dan aturan bisnis yang relevan.
