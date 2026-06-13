import warnings
# Menyembunyikan semua peringatan versi agar terminal bersih saat dijalankan
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# Konfigurasi halaman browser
st.set_page_config(page_title="GovTech - Rekomendasi Bansos", page_icon="🏛️", layout="centered")

# Judul Antarmuka Web
st.title("🏛️ Sistem Rekomendasi Program Bantuan Sosial")
st.write("Aplikasi analisis kelayakan bansos berdasarkan karakteristik mikro sosial-ekonomi rumah tangga.")
st.markdown("---")

# Mengamankan aset agar dimuat sekali saja (Caching)
@st.cache_resource
def load_assets():
    model_scaler = joblib.load('scaler_bansos.pkl')
    df_programs = pd.read_csv('df_programs.csv')
    return model_scaler, df_programs

try:
    model_scaler, df_programs = load_assets()
    features = ['is_anak', 'is_lansia', 'jml_anggota', 'total_pengeluaran', 'jenis_lantai', 'sumber_air']
    program_vectors = df_programs[features].values
except Exception as e:
    st.error(f"Gagal memuat file model. Pastikan 'scaler_bansos.pkl' dan 'df_programs.csv' berada di folder yang sama.")
    st.stop()

# --- INTERFACE FORM INPUT ---
st.subheader("📋 Form Input Data Rumah Tangga")

with st.form("form_bansos"):
    col1, col2 = st.columns(2)
    
    with col1:
        jml_anggota = st.number_input("Total Anggota Keluarga (Orang)", min_value=1, max_value=30, value=3)
        is_anak = st.number_input("Jumlah Anak Usia Sekolah (6-18 Tahun)", min_value=0, max_value=15, value=0)
        is_lansia = st.number_input("Jumlah Anggota Lansia (>60 Tahun)", min_value=0, max_value=15, value=0)
        
    with col2:
        total_pengeluaran = st.number_input("Estimasi Pengeluaran Bulanan Total (Rp)", min_value=0, value=750000, step=50000)
        
        jenis_lantai_label = st.selectbox("Kondisi Bahan Lantai Rumah", ["Semen / Tanah / Bambu Sederhana", "Ubin / Keramik / Marmer"])
        jenis_lantai = 1 if jenis_lantai_label == "Ubin / Keramik / Marmer" else 0
        
        sumber_air_label = st.selectbox("Sumber Air Minum Utama", ["Sumur / Sungai / Air Hujan / Eceran", "Ledeng (PDAM) / Air Kemasan Bermerek"])
        sumber_air = 1 if sumber_air_label == "Ledeng (PDAM) / Air Kemasan Bermerek" else 0

    # Tombol submit form
    submitted = st.form_submit_button("Analisis Kelayakan Bantuan")

# --- PROSES ENGINE REKOMENDASI SAAT TOMBOL DIKLIK ---
if submitted:
    # 1. Susun menjadi DataFrame sesuai format training populasi IFLS
    input_df = pd.DataFrame([[
        is_anak,
        is_lansia,
        jml_anggota,
        total_pengeluaran,
        jenis_lantai,
        sumber_air
    ]], columns=features)
    
    # 2. Transformasi Normalisasi skala data
    input_scaled = model_scaler.transform(input_df)
    
    # 3. Inversi nilai indikator ekonomi dan infrastruktur fisik
    input_scaled[0, 3] = 1 - input_scaled[0, 3]  # total_pengeluaran
    input_scaled[0, 4] = 1 - input_scaled[0, 4]  # jenis_lantai
    input_scaled[0, 5] = 1 - input_scaled[0, 5]  # sumber_air
    
    # 4. Hitung kemiripan dengan Cosine Similarity (Soft Ranker)
    scores = cosine_similarity(input_scaled, program_vectors)
    
    # 5. Mapping Hasil Awal
    results = []
    for i, prog_name in enumerate(df_programs['program']):
        results.append({
            'program': prog_name,
            'score': round(scores[0][i] * 100, 2)
        })
        
    # 6. Penerapan Regulasi Mutlak Hukum Bansos (Hard Filter / Knowledge-Based)
    for res in results:
        if res['program'] == 'KIP' and is_anak == 0:
            res['score'] = 0.0
        if res['program'] == 'PKH' and (is_anak == 0 and is_lansia == 0):
            res['score'] = 0.0
            
    # 7. Sorting berdasarkan ranking kecocokan tertinggi
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    # --- MENAMPILKAN HASIL RANKING & VISUALISASI PROGRESS BAR ---
    st.markdown("---")
    st.subheader("📊 Hasil Analisis & Peringkat Rekomendasi")
    
    for res in results:
        if res['program'] == 'BPNT':
            desc = "Bantuan Pangan Non-Tunai (Fokus pada Pemenuhan Kebutuhan Pangan Pokok Semabko)"
        elif res['program'] == 'PKH':
            desc = "Program Keluarga Harapan (Bantuan Bersyarat Komponen Pendidikan, Kesehatan, dan Lansia)"
        else:
            desc = "Kartu Indonesia Pintar (Bantuan Khusus Akses Biaya Pendidikan Anak Sekolah)"
            
        st.write(f"### **{res['program']}**")
        st.caption(desc)
        
        # Tampilan visual bar kelayakan
        st.progress(res['score'] / 100.0)
        st.write(f"Persentase Keseimbangan Kriteria: **{res['score']}%**")
        
        # Indikator Keputusan Akhir Sistem SPK Hybrid
        if res['score'] >= 80:
            st.success("💡 **Prioritas Utama**: Karakteristik sangat cocok. Rumah tangga diprioritaskan mendapat bantuan ini.")
        elif res['score'] > 0:
            st.warning("⚠️ **Prioritas Cadangan**: Memenuhi aspek ekonomi dasar, namun komponen pendukung kurang dominan.")
        else:
            st.error("❌ **Tidak Memenuhi Syarat**: Rumah tangga digugurkan oleh sistem karena melanggar regulasi dasar program.")
        st.markdown("")