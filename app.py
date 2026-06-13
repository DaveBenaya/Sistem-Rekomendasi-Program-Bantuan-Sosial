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

# 1. KONFIGURASI HALAMAN BROWSER
st.set_page_config(
    page_title="GovTech Analytics - Bansos",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. INJEKSI KUSTOM CSS (Auto-Adaptive Theme System)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;600;700&display=swap');
    
    :root {
        --bg-main: #ffffff;
        --bg-sidebar: #f8fafc;
        --text-main: #1e293b;
        --text-muted: #64748b;
        --bg-card: #ffffff;
        --border-card: #e2e8f0;
        --shadow-hero: rgba(212, 175, 55, 0.12);
        --shadow-card: rgba(0, 0, 0, 0.04);
    }
    
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-main: #1e2538;
            --bg-sidebar: #151a26;
            --text-main: #ffffff;
            --text-muted: #94a3b8;
            --bg-card: #283149;
            --border-card: #3b4766;
            --shadow-hero: rgba(212, 175, 55, 0.25);
            --shadow-card: rgba(0, 0, 0, 0.2);
        }
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Inter', sans-serif;
        background-color: var(--bg-main) !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-card);
    }
    
    .hero-container {
        background: var(--bg-card);
        padding: 2.5rem;
        border-radius: 16px;
        border: 2px solid #d4af37;
        box-shadow: 0 4px 20px var(--shadow-hero);
        margin-bottom: 2rem;
    }
    .hero-title { margin:0; color: var(--text-main); font-size: 2.3rem; font-weight:700; }
    .hero-subtitle { margin:8px 0 0 0; color: var(--text-muted); font-size:1.1rem; }
    
    .custom-card {
        background: var(--bg-card);
        border: 1px solid var(--border-card);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 8px var(--shadow-card);
        transition: all 0.2s ease;
    }
    .custom-card:hover {
        transform: translateY(-2px);
        border-color: #d4af37;
        box-shadow: 0 6px 16px var(--shadow-hero);
    }
    .card-title { margin:0; color: var(--text-main); font-size:1.3rem; }
    .card-desc { margin: 6px 0 12px 0; color: var(--text-muted); font-size:0.9rem; }
    
    .sidebar-title { text-align: center; margin-top:0; color: var(--text-main); }
    
    .badge {
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
    .badge-prio { background-color: #fef9c3; color: #854d0e; border: 1px solid #fef08a; }
    .badge-cadangan { background-color: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
    .badge-gagal { background-color: #fef2f2; color: #991b1b; border: 1px solid #fee2e2; }
    
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #d4af37 0%, #b45309 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.6rem 2rem !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(212, 175, 55, 0.3) !important;
        width: 100%;
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(212, 175, 55, 0.5) !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. LOAD ASSET MODEL
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

# 4. SIDEBAR COMPONENTS
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #d4af37; font-size: 3.5rem; margin-bottom:0;'>🏛️</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='sidebar-title'>GovTech Engine</h3>", unsafe_allow_html=True)
    st.caption("Sistem Pendukung Keputusan Kelayakan Penerima Bantuan Sosial Nasional.")
    st.markdown("---")
    st.markdown("💡 **Metode Sistem:**")
    st.info("Hybrid Architecture\n1. Knowledge-Based (Filter)\n2. Content-Based (Ranking)")
    st.markdown("📂 **Basis Data Baseline:**")
    st.code("IFLS-5 (RAND Corp)")

# 5. HERO SECTION UTAMA
st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">🏛️ Analytics Dashboard & Recommendation System</h1>
        <p class="hero-subtitle">Optimasi Akurasi Alokasi Bantuan Sosial Berbasis Karakteristik Rumah Tangga Mikro</p>
    </div>
""", unsafe_allow_html=True)

# 6. PEMBAGIAN LAYOUT KOLOM UTAMA
left_column, right_column = st.columns([1.1, 1.3], gap="large")

with left_column:
    st.markdown("### 📋 Parameter Input Keluarga")
    st.write("Silakan lengkapi profil sosiologis dan finansial di bawah ini:")
    
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

# 7. LOGIKA ENGINE DAN VISUALISASI OUTPUT DENGAN PENJELASAN LOGIS
with right_column:
    st.markdown("### 📊 Hasil Komputasi Kelayakan")
    
    if btn_click:
        input_df = pd.DataFrame([[is_anak, is_lansia, jml_anggota, total_pengeluaran, jenis_lantai, sumber_air]], columns=features)
        input_scaled = model_scaler.transform(input_df)
        
        input_scaled[0, 3] = 1 - input_scaled[0, 3]
        input_scaled[0, 4] = 1 - input_scaled[0, 4]
        input_scaled[0, 5] = 1 - input_scaled[0, 5]
        
        scores = cosine_similarity(input_scaled, program_vectors)
        
        results = []
        for i, prog_name in enumerate(df_programs['program']):
            results.append({'program': prog_name, 'score': round(scores[0][i] * 100, 2)})
            
        # Variabel penampung simpulan akhir
        program_diterima = []
        program_ditolak = []
            
        # Proses Filter Hukum & Penentuan Narasi Alasan Kustom
        for res in results:
            alasan = ""
            
            # ATURAN KIP
            if res['program'] == 'KIP':
                if is_anak == 0:
                    res['score'] = 0.0
                    alasan = "❌ **Diskualifikasi Mutlak**: Regulasi kementerian menetapkan program KIP hanya diperuntukkan bagi keluarga yang memiliki anak usia sekolah aktif (6-18 tahun)."
                    program_ditolak.append("KIP (Tidak ada komponen anak sekolah)")
                elif res['score'] >= 80:
                    alasan = f"🔹 **Diterima (Prioritas Utama)**: Rumah tangga memiliki {is_anak} anak sekolah dan kondisi ekonomi berada di batas bawah populasi, sangat membutuhkan subsidi biaya pendidikan."
                    program_diterima.append("KIP")
                else:
                    alasan = "🔸 **Dipertimbangkan (Cadangan)**: Memiliki komponen anak sekolah, namun tingkat pengeluaran bulanan dinilai masih cukup mampu mandiri dibandingkan target klaster miskin ekstrem."
                    program_diterima.append("KIP (Cadangan)")
            
            # ATURAN PKH
            elif res['program'] == 'PKH':
                if is_anak == 0 and is_lansia == 0:
                    res['score'] = 0.0
                    alasan = "❌ **Diskualifikasi Mutlak**: Aturan hukum PKH mewajibkan adanya minimal salah satu komponen sensitif, yaitu anak usia sekolah atau anggota keluarga lanjut usia."
                    program_ditolak.append("PKH (Tidak memiliki komponen anak/lansia)")
                elif res['score'] >= 80:
                    alasan = f"🔹 **Diterima (Prioritas Utama)**: Kondisi sosial sangat mendesak dengan adanya komponen tanggungan ({is_anak} anak, {is_lansia} lansia) dikombinasikan dengan keterbatasan finansial dan fasilitas rumah."
                    program_diterima.append("PKH")
                else:
                    alasan = "🔸 **Dipertimbangkan (Cadangan)**: Memiliki komponen bersyarat yang sah, namun indeks kemiskinan gabungan (ekonomi & rumah) belum masuk dalam zona prioritas utama."
                    program_diterima.append("PKH (Cadangan)")
            
            # ATURAN BPNT
            elif res['program'] == 'BPNT':
                if res['score'] >= 80:
                    alasan = "🔹 **Diterima (Prioritas Utama)**: Indikator ekonomi total pengeluaran dan buruknya fasilitas fisik rumah (lantai/air) menempatkan keluarga ini sebagai prioritas tinggi penerima jaminan pangan pokok."
                    program_diterima.append("BPNT")
                else:
                    alasan = "🔸 **Dipertimbangkan (Cadangan)**: Kondisi ekonomi berada di zona tengah (ambang batas). Bantuan pangan baru akan dialokasikan jika kuota kuantum utama daerah masih tersedia."
                    program_diterima.append("BPNT (Cadangan)")

            # Penentuan Desain Badge
            if res['score'] >= 80:
                badge_style = '<span class="badge badge-prio">Prioritas Utama</span>'
            elif res['score'] > 0:
                badge_style = '<span class="badge badge-cadangan">Prioritas Cadangan</span>'
            else:
                badge_style = '<span class="badge badge-gagal">Sistem Diskualifikasi</span>'
            
            # Render Tampilan Kartu Hasil Per Program
            st.markdown(f"""
                <div class="custom-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 class="card-title">Program {res['program']}</h4>
                        {badge_style}
                    </div>
                    <p class="card-desc">{desc if 'desc' in locals() else ""}</p>
                    <div style="font-size: 0.9rem; padding: 10px; border-radius: 6px; background: rgba(212,175,55,0.05); border-left: 3px solid #d4af37;">
                        {alasan}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.progress(res['score'] / 100.0)
            st.write(f"Match Core Score: **{res['score']}%**")
            st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
            
        # 8. SECTION KESIMPULAN AKHIR INTEGRATIF (BARU)
        st.markdown("---")
        st.markdown("### 📌 Ringkasan & Rekomendasi Akhir")
        
        with st.container(border=True):
            st.write("**Bantuan yang BISA diajukan (Prioritas):**")
            if program_diterima:
                for p in program_diterima:
                    st.markdown(f"✅ `{p}`")
            else:
                st.markdown("ℹ️ *Tidak ada bantuan prioritas yang cocok dengan profil saat ini.*")
                
            st.write("**Bantuan yang TIDAK BISA diterima:**")
            if program_ditolak:
                for p in program_ditolak:
                    st.markdown(f"🚫 `{p}`")
            else:
                st.markdown("ℹ️ *Semua program lolos kriteria penyaringan regulasi hukum dasar.*")
                
            st.caption("💡 *Catatan Teknis: Rekomendasi disusun secara otomatis oleh SPK Hybrid menggunakan basis data baseline makro IFLS-5 (RAND Corporation) berstandar nasional.*")
            
    else:
        st.info("💡 Menunggu Eksekusi: Silakan sesuaikan data keluarga di panel kiri, kemudian klik tombol 'PROSES ENGINE REKOMENDASI' untuk memunculkan matriks peringkat kelayakan bansos.")