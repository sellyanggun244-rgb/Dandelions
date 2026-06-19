import os
import numpy as np
import streamlit as stream
from PIL import Image
import tempfile

# IMPOR ENGINE ASLI
import model.eigenface as ef

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def hitung_pca_core(matrix, n_comp):
    h, w = matrix.shape
    rata_rata = np.mean(matrix, axis=0)
    centered = matrix - rata_rata
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    n = min(n_comp, len(S), w, h)
    S_diag = np.diag(S[:n])
    reconstructed = np.dot(U[:, :n], np.dot(S_diag, Vt[:n, :])) + rata_rata
    return np.clip(reconstructed, 0, 255).astype(np.uint8)

if not ef.model.is_trained:
    ef.run_training_manually()

# TAMPILAN ANTARMUKA STREAMLIT
stream.set_page_config(page_title="FaceVerifier — DeepFace & PCA", layout="wide")

stream.markdown("""
<style>

.main {
    background-color: #FAF8F5;
}

.block-container{
    padding-top:2rem;
    max-width:1200px;
}

[data-testid="stSidebar"]{
    background:#F7F4F1;
}

h1,h2,h3,h4{
    color:#2D3748;
}

.stButton button{
    background:#7C5C4E;
    color:white;
    border:none;
    border-radius:10px;
    height:45px;
    font-weight:600;
}

.stButton button:hover{
    background:#6B4D40;
}

.stProgress > div > div > div > div{
    background:#7C5C4E;
}

</style>
""", unsafe_allow_html=True)

with stream.sidebar:
    stream.markdown('<h1 style="font-size: 32px; font-weight: 700; color: #2D3748; letter-spacing: -0.5px;">Face<span style="color: #7C5C4E;">Verifier</span></h1>', unsafe_allow_html=True)
    stream.markdown(
    """
    <div style="
    font-size:18px;
    font-weight:600;
    color:#2D3748;
    margin-bottom:8px;
    ">
    Face Similarity Analysis System
    </div>
    """,
    unsafe_allow_html=True
    )
    
    stream.markdown(
        '<p style="font-size: 13.5px; color: #6B7280; line-height: 1.7;">'
        'Sistem verifikasi wajah untuk membandingkan dua citra menggunakan FaceNet dan Principal Component Analysis (PCA).'
        '</p>', 
        unsafe_allow_html=True
    )
    with stream.container(border=True):
        stream.markdown("**FaceNet + PCA Analysis Engine**")
        stream.markdown(
            '<p style="font-size:12.5px; color:#6B7280; line-height:1.6; margin:0; margin-bottom:10px;">'
            'Ekstraksi fitur biometrik wajah dan reduksi dimensi untuk analisis kemiripan yang efisien.'
            '</p>',
            unsafe_allow_html=True
        )
    
    stream.markdown("<br>", unsafe_allow_html=True)
    stream.markdown("### Fitur Tambahan")
    with stream.container(border=True):
        stream.markdown("**PCA Image Compression**")
        n_components_slider = stream.slider("Pilih Komponen Utama (PCA)", 5, 150, 50)
stream.markdown('<h2 style="font-size:26px; font-weight:700; color:#2D3748;">Verifikasi Kemiripan Wajah</h2>', unsafe_allow_html=True)

stream.markdown(
    '<p style="font-size:14.5px; color:#6B7280; line-height:1.6; margin-bottom: 25px;">'
    'Unggah dua foto wajah (JPG/PNG) dan mulai proses autentikasi. '
    '</p>', 
    unsafe_allow_html=True
)

col1, col2 = stream.columns(2)

with col1:
    stream.markdown(
        '<div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">'
        '<span style="background: #EEE7DF; color: #6B4D40; font-size: 11px; font-weight: 700; padding: 4px 8px; border-radius: 6px;">Sampel A</span>'
        '<span style="font-size: 14px; font-weight: 600; color: #374151;">Foto Masa Kecil</span>'
        '</div>', 
        unsafe_allow_html=True
    )
    file_baby = stream.file_uploader("Pilih dokumen citra wajah A", type=["jpg", "jpeg", "png"], key="baby", label_visibility="collapsed")
    if file_baby:
        stream.markdown('<p style="font-size:13px; color:#15803d; font-weight:500; margin-top:-10px;">✓ Foto berhasil diunggah</p>', unsafe_allow_html=True)
        img_a = Image.open(file_baby)
        stream.image(img_a, caption="Pratinjau Foto Masa Kecil", use_container_width=True)

with col2:
    stream.markdown(
        '<div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">'
        '<span style="background: #EEE7DF; color: #6B4D40; font-size: 11px; font-weight: 700; padding: 4px 8px; border-radius: 6px;">Sampel B</span>'
        '<span style="font-size: 14px; font-weight: 600; color: #374151;">Foto Sekarang</span>'
        '</div>', 
        unsafe_allow_html=True
    )
    file_adult = stream.file_uploader("Pilih dokumen citra wajah B", type=["jpg", "jpeg", "png"], key="adult", label_visibility="collapsed")
    if file_adult:
        stream.markdown('<p style="font-size:13px; color:#15803d; font-weight:500; margin-top:-10px;">✓ Foto berhasil diunggah</p>', unsafe_allow_html=True)
        img_b = Image.open(file_adult)
        stream.image(img_b, caption="Pratinjau Foto Sekarang", use_container_width=True)

stream.markdown("---")

if stream.button("Mulai Verifikasi", type="primary", use_container_width=True):
    if not file_baby or not file_adult:
        stream.error("⚠️ Gagal Mengevaluasi: Silakan pilih kedua dokumen citra wajah terlebih dahulu.")
    else:
        tmp_dir = tempfile.gettempdir()
        tmp_baby_path = os.path.join(tmp_dir, "tmp_face1.png")
        tmp_adult_path = os.path.join(tmp_dir, "tmp_face2.png")
        
        Image.open(file_baby).save(tmp_baby_path)
        Image.open(file_adult).save(tmp_adult_path)
        
        try:
            with stream.spinner("Mengekstraksi fitur wajah dan melakukan proyeksi PCA..."):
                res = ef.compare_faces(tmp_baby_path, tmp_adult_path)
            
            stream.markdown("""
            <div style="
            background:white;
            padding:20px;
            border-radius:18px;
            border:1px solid #E5E7EB;
            box-shadow:0 4px 12px rgba(0,0,0,0.04);
            ">
            """, unsafe_allow_html=True)
            
            with stream.container(border=True):
                stream.markdown('<h4 style="font-size:15px; font-weight:700; color:#2F3437; margin-bottom:12px;">Hasil Analisis Biometrik Wajah</h4>', unsafe_allow_html=True)
                
                is_match = "Sama" in res["result"]
                if is_match:
                    stream.markdown(f'<span style="background:#dcfce7; color:#15803d; font-size:13px; font-weight:700; padding:6px 12px; border-radius:6px; display:inline-block; margin-bottom:15px;">{res["result"]}</span>', unsafe_allow_html=True)
                else:
                    stream.markdown(f'<span style="background:#FEF3C7; color:#B45309; font-size:13px; font-weight:700; padding:6px 12px; border-radius:6px; display:inline-block; margin-bottom:15px;">{res["result"]}</span>', unsafe_allow_html=True)
                
                try:
                    val_progress = float(res["similarity"]) / 100.0
                    stream.progress(min(max(val_progress, 0.0), 1.0))
                except:
                    stream.progress(0.5)
                
                stream.markdown('<div style="margin-top:15px;"></div>', unsafe_allow_html=True)
                m_col1, m_col2, m_col3, m_col4 = stream.columns(4)
                
                with m_col1:
                    with stream.container(border=True):
                        stream.markdown('<p style="font-size:11px; color:#7A7A7A; margin:0;">Cosine Similarity</p>', unsafe_allow_html=True)
                        stream.markdown(f'<strong style="font-size:13px; color:#2F3437;">{res["cosine"]}</strong>', unsafe_allow_html=True)
                with m_col2:
                    with stream.container(border=True):
                        stream.markdown('<p style="font-size:11px; color:#7A7A7A; margin:0;">Euclidean Distance</p>', unsafe_allow_html=True)
                        stream.markdown(f'<strong style="font-size:13px; color:#2F3437;">{res["euclidean"]}</strong>', unsafe_allow_html=True)
                with m_col3:
                    with stream.container(border=True):
                        stream.markdown('<p style="font-size:11px; color:#7A7A7A; margin:0;">Ambang Batas (Threshold)</p>', unsafe_allow_html=True)
                        stream.markdown('<strong style="font-size:13px; color:#2F3437;">0.4500</strong>', unsafe_allow_html=True)
                
                with m_col4:
                    with stream.container(border=True):
                        stream.markdown(
                            '<p style="font-size:11px; color:#7A7A7A; margin:0;">Kemiripan</p>',
                            unsafe_allow_html=True
                        )
                        stream.markdown(
                            f'''
                            <div style="
                            text-align:center;
                            font-size:26px;
                            font-weight:700;
                            color:#7C5C4E;
                            ">
                            {res["similarity"]}%
                            </div>
                            ''',
                            unsafe_allow_html=True
                        )
            
            stream.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as e:
            stream.error(f"⚠️ Gagal Mengevaluasi: {str(e)}")
        finally:
            if os.path.exists(tmp_baby_path): os.remove(tmp_baby_path)
            if os.path.exists(tmp_adult_path): os.remove(tmp_adult_path)

if stream.button("Mulai Ulang", type="secondary", use_container_width=True):
    for key in list(stream.session_state.keys()):
        del stream.session_state[key]
    stream.rerun()

# Simulasi Kompresi Citra dengan PCA CITRA TUNGGAL (Indentasi Sudah Diperbaiki Ke Tepi Kiri)
stream.markdown("---")
stream.subheader("Simulasi Kompresi Citra dengan PCA")

pilihan_sampel = []
if file_baby: pilihan_sampel.append("Sampel A (Foto Masa Kecil)")
if file_adult: pilihan_sampel.append("Sampel B (Foto Sekarang)")

pilihan_terpilih = stream.radio("Pilih citra untuk dianalisis:", pilihan_sampel, horizontal=True)

if stream.button("Proses Kompresi Citra"):
    file_target = file_baby if "Sampel A" in pilihan_terpilih else file_adult
    nama_caption = "Sampel A" if "Sampel A" in pilihan_terpilih else "Sampel B"
    
    img_gray = Image.open(file_target).convert('L')
    arr_gray = np.array(img_gray)
    res_gray = hitung_pca_core(arr_gray, n_components_slider)
    
    c_gray1, c_gray2 = stream.columns(2)
    with c_gray1:
        stream.image(img_gray, caption=f"Citra Grayscale Asli ({nama_caption})", use_container_width=True)
    with c_gray2:
        stream.image(res_gray, caption=f"Hasil Rekonstruksi PCA ({n_components_slider} Komponen)", use_container_width=True)