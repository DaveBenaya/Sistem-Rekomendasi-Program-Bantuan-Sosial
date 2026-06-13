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

# 1. KONFIGURASI HALAMAN BROWSER (Modern Dark Theme Basic)
st.set_page_config(
    page_title="GovTech Analytics - Bansos",
    page_icon="🏛️",
    layout="wide", # Diubah ke wide agar layout terlihat seperti dashboard luas
    initial_sidebar_state="expanded"
)

# 2. INJEKSI KUSTOM CSS (Kunci Tampilan Kekinian & Elegan)
st.markdown("""
    <style>
    /* Mengubah font global dan background dashboard */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero Section Gradient Box */
    .hero-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        padding: 2.5rem;
        border-radius: 16px;
        border: 1px solid #312e81;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
    }
    
    /* Kustomisasi Kartu Rekomendasi (Glassmorphism effect) */
    .custom-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .custom-card:hover {
        transform: translateY(-2px);
        border-color: #6366f1;
    }
    
    /* Badge Status */
    .badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
    .badge-prio { background-color: #1e1b4b; color: #a5b4fc; border: 1px solid #4338ca; }
    .badge-cadangan { background-color: #312e81; color: #fed7aa; border: 1px solid #7c2d12; }
    .badge-gagal { background-color: #451a03; color: #fca5a5; border: 1px solid #991b1b; }
    
    /* Tombol Analisis Modern */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.6rem 2rem !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
        width: 100%;
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. LOAD ASSET MODEL MODEL DENGAN CACHING AMAN
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
    st.error("Gagal memuat file konfigurasi model backend.")
    st.stop()

# 4. SIDEBAR METADATA INFORMASI PROJEK
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/000000/government-building.png", width=70)
    st.markdown("### **GovTech Engine**")
    st.caption("Sistem Pendukung Keputusan Kelayakan Penerima Bantuan Sosial Nasional.")
    st.markdown("---")
    
    # Menampilkan informasi akademik secara estetis
    st.markdown("💡 **Metode Sistem:**")
    st.info("Hybrid Architecture\n1. Knowledge-Based (Filter)\n2. Content-Based (Ranking)")
    
    st.markdown("📂 **Basis Data Baseline:**")
    st.code("IFLS-5 (RAND Corp)")

# 5. HERO SECTION UTAMA
st.markdown("""
    <div class="hero-container">
        <h1 style='margin:0; color:#ffffff; font-size: 2.3rem; font-weight:700;'>🏛️ Analytics Dashboard & Recommendation System</h1>
        <p style='margin:8px 0 0 0; color:#94a3b8; font-size:1.1rem;'>Optimasi Akurasi Alokasi Bantuan Sosial Berbasis Karakteristik Rumah Tangga Mikro</p>
    </div>
""", unsafe_allow_html=True)

# 6. PEMBAGIAN LAYOUT KOLOM UTAMA (Kiri: Form Input, Kanan: Hasil Real-time)
left_column, right_column = st.columns([1.1, 1.3], gap="large")

with left_column:
    st.markdown("### 📋 Parameter Input Keluarga")
    st.write("Silakan lengkapi profil sosiologis dan finansial di bawah ini:")
    
    # Form dibungkus wadah rapi
    with st.container(border=True):
        jml_anggota = st.slider("Total Anggota Keluarga (Orang)", 1, 15, 4)
        
        col_slot1, col_slot2 = st.columns(2)
        with col_slot1:
            is_anak = st.number_input("Anak Usia Sekolah (6-18 Thn)", min_value=0, max_value=10, value=1)
        with col_slot2:
            is_lansia = st.number_input("Anggota Lansia (>60 Thn)", min_value=0, max_value=10, value=0)
            
        total_pengeluaran = st.number_input("Total Pengeluaran Bulanan (Rp)", min_value=0, value=1200000, step=100000)
        
        st.markdown("**Kondisi Infrastruktur Fisik Rumah:**")
        jenis_lantai_label = st.radio("Material Lantai Terluas", ["Tanah / Semen Kasar / Bambu Sederhana", "Ubin / Keramik / Marmer"], horizontal=True)
        jenis_lantai = 1 if "Keramik" in jenis_lantai_label else 0
        
        sumber_air_label = st.radio("Akses Sumber Air Minum", ["Sumur / Sungai / Air Hujan / Eceran", "PDAM Ledeng / Air Kemasan Bermerek"], horizontal=True)
        sumber_air = 1 if "PDAM" in sumber_air_label else 0
        
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        btn_click = st.button("PROSES ENGINE REKOMENDASI")

# 7. LOGIKA ENGINE DAN VISUALISASI OUTPUT DI KOLOM KANAN
with right_column:
    st.markdown("### 📊 Hasil Komputasi Kelayakan")
    
    if btn_click:
        # Pipeline Pemrosesan Data
        input_df = pd.DataFrame([[is_anak, is_lansia, jml_anggota, total_pengeluaran, jenis_lantai, sumber_air]], columns=features)
        input_scaled = model_scaler.transform(input_df)
        
        # Inversi Nilai Indikator (Kemiskinan = Nilai Skala Tinggi)
        input_scaled[0, 3] = 1 - input_scaled[0, 3]
        input_scaled[0, 4] = 1 - input_scaled[0, 4]
        input_scaled[0, 5] = 1 - input_scaled[0, 5]
        
        # Penghitungan Kemiripan
        scores = cosine_similarity(input_scaled, program_vectors)
        
        results = []
        for i, prog_name in enumerate(df_programs['program']):
            results.append({'program': prog_name, 'score': round(scores[0][i] * 100, 2)})
            
        # Penerapan Regulasi Mutlak (Hard Filter)
        for res in results:
            if res['program'] == 'KIP' and is_anak == 0:
                res['score'] = 0.0
            if res['program'] == 'PKH' and (is_anak == 0 and is_lansia == 0):
                res['score'] = 0.0
                
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        # Menampilkan Kartu Visual secara Interaktif
        for res in results:
            if res['program'] == 'BPNT':
                desc = "Fokus penjaminan pangan pokok sembako keluarga pra-sejahtera."
                badge_style = '<span class="badge badge-prio">Prioritas Utama</span>' if res['score'] >= 80 else ('<span class="badge badge-cadangan">Prioritas Cadangan</span>' if res['score'] > 0 else '<span class="badge badge-gagal">Sistem Diskualifikasi</span>')
            elif res['program'] == 'PKH':
                desc = "Bantuan bersyarat untuk klaster pendidikan, kesehatan anak, dan kesejahteraan lansia."
                badge_style = '<span class="badge badge-prio">Prioritas Utama</span>' if res['score'] >= 80 else ('<span class="badge badge-cadangan">Prioritas Cadangan</span>' if res['score'] > 0 else '<span class="badge badge-gagal">Sistem Diskualifikasi</span>')
            else:
                desc = "Kompensasi tunai khusus alokasi biaya keberlanjutan sekolah anak."
                badge_style = '<span class="badge badge-prio">Prioritas Utama</span>' if res['score'] >= 80 else ('<span class="badge badge-cadangan">Prioritas Cadangan</span>' if res['score'] > 0 else '<span class="badge badge-gagal">Sistem Diskualifikasi</span>')
            
            # HTML Injeksi Kartu Kustom
            st.markdown(f"""
                <div class="custom-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin:0; color:#ffffff; font-size:1.3rem;">Program {res['program']}</h4>
                        {badge_style}
                    </div>
                    <p style="margin: 6px 0 12px 0; color:#94a3b8; font-size:0.9rem;">{desc}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Menampilkan Nilai Skor dengan Bar Bawaan Streamlit agar Tetap Sinkron
            st.progress(res['score'] / 100.0)
            st.write(f"Match Core Score: **{res['score']}%**")
            st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
            
    else:
        # Tampilan Awal Sebelum User Klik Tombol Rekomendasi
        st.info("💡 Menunggu Eksekusi: Silakan sesuaikan data keluarga di panel kiri, kemudian klik tombol 'PROSES ENGINE REKOMENDASI' untuk memunculkan matriks peringkat kelayakan bansos.")