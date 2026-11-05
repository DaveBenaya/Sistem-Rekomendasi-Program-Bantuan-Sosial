# 🏛️ Sistem Rekomendasi Program Bantuan Sosial Berbasis Karakteristik Rumah Tangga Mikro

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Sistem Pendukung Keputusan (*Decision Support System*) berbasis kecerdasan buatan untuk mengoptimalkan ketepatan sasaran dan personalisasi alokasi bantuan sosial nasional (PKH, BPNT, KIP) menggunakan arsitektur **Hybrid Recommendation System (Knowledge-Based Hard Filtering + Content-Based Filtering Soft Ranking)**.

---

## 📌 Latar Belakang

* **Tumpang Tindih & Duplikasi Alokasi:** Pendataan bansos konvensional sering mengalami tumpang tindih alokasi antar-program kementerian di tingkat daerah.
* **Bias Inklusi & Eksklusi (*Targeting Errors*):** Tingginya angka salah sasaran di mana keluarga mampu menerima bantuan sementara keluarga rentan justru tereleminasi akibat bias pendataan statis.
* **Kebutuhan Personalisasi Profil Mikro:** Penentuan kelayakan membutuhkan pemrosesan cerdas multi-indikator yang mencakup demografi keluarga, daya beli riil, hingga kondisi sanitasi dan fisik hunian.

---

## 🎯 Tujuan & Manfaat

* **Personalisasi Bantuan Sosial:** Mengidentifikasi program bantuan (Program Keluarga Harapan/PKH, Bantuan Pangan Non-Tunai/BPNT, dan Kartu Indonesia Pintar/KIP) yang paling sesuai dengan komposisi kebutuhan keluarga.
* **Efisiensi Anggaran Negara:** Meminimalkan kebocoran fiskal akibat *inclusion error*.
* **Transparansi & *Explainable AI*:** Memberikan transparansi aturan (*rule justification*) dan persentase skor kecocokan (*match core score*) kepada masyarakat maupun petugas lapangan kementerian.

---

## ⚙️ Metodologi & Arsitektur Sistem

Sistem ini menerapkan pendekatan **Hybrid Filtering** dua lapis untuk menyeimbangkan kepatuhan regulasi hukum negara dan fleksibilitas perankingan data:

```
[Input Data Keluarga]
         │
         ▼
[Layer 1: Knowledge-Based Hard Filter]
   ├── Global Financial Threshold (Rp 4.500.000)
   ├── KIP Requirement (Wajib ada anak usia 6-18 thn)
   └── PKH Requirement (Wajib ada anak sekolah / lansia >60 thn)
         │
         ├─► [Diskualifikasi Mutlak (Score: 0%)]
         ▼ (Lolos Regulasi)
[Layer 2: Content-Based Filtering Soft Ranker]
   ├── MinMax Normalization & Inversi Fitur Ekonomi
   ├── Cosine Similarity Calculation vs Program Profiles
   └── Priority Leveling (Prioritas Utama vs Cadangan)
         │
         ▼
[Dashboard Rekomendasi & Justifikasi Aturan]
```

### 1. Knowledge-Based Hard Filter (Lapisan Regulasi)
* **Poverty Threshold (Batas Mampu Global):** Pengeluaran bulanan $\ge \text{Rp } 4.500.000$ secara otomatis mendiskualifikasi seluruh program bantuan.
* **Regulasi KIP:** Wajib memiliki minimal satu anak sekolah (`is_anak > 0`).
* **Regulasi PKH:** Wajib memiliki minimal salah satu komponen rentan (`is_anak > 0` atau `is_lansia > 0`).

### 2. Content-Based Filtering (Lapisan Peringkat Kemiripan)
Menghitung kedekatan vektor karakteristik keluarga pemohon ($U$) terhadap kriteria ideal program bansos ($I$) menggunakan metrik **Cosine Similarity**:

$$\text{Similarity}(U, I) = \frac{\sum_{i=1}^{n} U_i I_i}{\sqrt{\sum_{i=1}^{n} U_i^2} \sqrt{\sum_{i=1}^{n} I_i^2}}$$

* **Prioritas Utama:** $\text{Match Score} \ge 75\%$
* **Prioritas Cadangan:** $0\% < \text{Match Score} < 75\%$
* **Sistem Diskualifikasi:** $\text{Match Score} = 0\%$

---

## 📊 Dataset & Rekayasa Fitur (*Feature Engineering*)

Model dilatih menggunakan basis data mikro skala nasional **Indonesian Family Life Survey Wave 5 (IFLS-5)** yang dirilis oleh RAND Corporation (mencakup 15.921 observasi rumah tangga):
* **Buku BK_AR1:** Data roster demografi (ekstraksi usia anak sekolah dan lansia).
* **Buku B1_KS1:** Data konsumsi dan total pengeluaran bulanan (agregasi variabel finansial).
* **Buku B2_KR:** Karakteristik fisik hunian (material lantai dan sanitasi sumber air).

| Fitur | Deskripsi | Transformasi |
| :--- | :--- | :--- |
| `is_anak` | Jumlah anak usia sekolah (6–18 tahun) | Agregasi Roster |
| `is_lansia` | Jumlah anggota lansia (>60 tahun) | Agregasi Roster |
| `jml_anggota` | Total anggota rumah tangga | Count Roster |
| `total_pengeluaran` | Pengeluaran konsumsi bulanan | $\log(1 + x) \rightarrow \text{MinMaxScaler} \rightarrow (1 - \text{scaled})$ |
| `jenis_lantai` | Kondisi fisik lantai (bambu/tanah vs ubin/keramik) | Biner $\rightarrow (1 - \text{scaled})$ |
| `sumber_air` | Akses sanitasi air (sumur/sungai vs PDAM/kemasan) | Biner $\rightarrow (1 - \text{scaled})$ |

> *Catatan Teknis:* Fitur ekonomi dan hunian diinversi ($1 - \text{scaled}$) sehingga nilai yang mendekati $1.0$ mencerminkan kerentanan sosial-ekonomi yang lebih tinggi.

---

## 📈 Evaluasi & Hasil Validasi

Berdasarkan pengujian simulasi berlabel keputusan manual kementerian/ahli:
* **Precision@1:** **91%**, menunjukkan ketepatan rekomendasi program teratas terhadap regulasi kementerian.
* **Solusi Cold-Start:** Pendekatan hybrid berhasil mengatasi kendala *cold-start* bagi rumah tangga pendaftar baru tanpa rekam jejak bansos historis.
* **Sensitivitas Kriteria:**
  * Profil Miskin Ekstrem terbukti memprioritaskan program BPNT (alokasi sembako & ketahanan pangan).
  * Profil Keluarga Tanggungan Lengkap secara konsisten memprioritaskan PKH dan KIP.

---

## 📁 Struktur Repositori

```text
├── app.py                      # Dashboard interaktif web (Streamlit)
├── notebook_training.ipynb     # Pipeline ETL data IFLS-5 & pembuatan model
├── df_programs.csv             # Matriks profil kriteria ideal program bansos
├── scaler_bansos.pkl           # Objek serialisasi MinMaxScaler
├── requirements.txt            # Daftar dependensi modul Python
├── .gitignore                  # Konfigurasi pengabaian file video/cache
└── README.md                   # Dokumentasi teknis proyek
```

---

## 🚀 Panduan Instalasi & Penggunaan

### 1. Kloning Repositori
```bash
git clone https://github.com/DaveBenaya/Sistem-Rekomendasi-Program-Bantuan-Sosial.git
cd Sistem-Rekomendasi-Program-Bantuan-Sosial
```

### 2. Buat Lingkungan Virtual (Opsional)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Pasang Dependensi
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi Dashboard
```bash
streamlit run app.py
```
Akses aplikasi melalui peramban web di `http://localhost:8501`.

---

## 📚 Referensi

1. RAND Corporation. (2015). *Indonesian Family Life Survey 5 (IFLS-5)*.
2. Ricci, F., Rokach, L., & Shapira, B. (Eds.). (2015). *Recommender Systems Handbook*. Springer.
3. Dwivedi, Y. K., et al. (2021). *Artificial Intelligence (AI): Multidisciplinary perspectives on emerging challenges, opportunities, and agenda for research, practice and policy*. International Journal of Information Management.

---

## 👤 Pengembang Proyek

* **Nama:** Dave Benaya Walujo (NRP: 3324600025)
* **Program Studi:** Sains Data Terapan, Politeknik Elektronika Negeri Surabaya (PENS)
* **Dosen Pengampu:** Ronny Susetyoko, S.Si., M.Si.
