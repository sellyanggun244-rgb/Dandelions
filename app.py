import os
import numpy as np
import streamlit as stream
from PIL import Image
import tempfile

import model.eigenface as ef

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


MAX_DIM_PCA = 600


def hitung_pca_core(matrix, n_comp):
    h, w = matrix.shape
    rata_rata = np.mean(matrix, axis=0)
    centered = matrix - rata_rata
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    n = max(min(n_comp, len(S), w, h), 1)
    S_diag = np.diag(S[:n])
    reconstructed = np.dot(U[:, :n], np.dot(S_diag, Vt[:n, :])) + rata_rata
    return np.clip(reconstructed, 0, 255).astype(np.uint8)


def resize_untuk_pca(img):
    w, h = img.size
    if max(w, h) > MAX_DIM_PCA:
        skala = MAX_DIM_PCA / max(w, h)
        img = img.resize((int(w * skala), int(h * skala)))
    return img


if not ef.model.is_trained:
    ef.run_training_manually()

stream.set_page_config(page_title="FaceVerifier — InsightFace & PCA", layout="wide")

stream.markdown("""
<style>
.stApp {
    background-color: #F4F1EC !important;
}
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
    background-color: #F8F6F2 !important;
    border-radius: 16px;
    padding-bottom: 2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.01);
}

[data-testid="stSidebar"] {
    background-color: #E6DDD4 !important;
    border-right: 1px solid #D6CDC3 !important;
}
[data-testid="stSidebarCollapseButton"] {
    background-color: #E6DDD4 !important;
}

h1, h2, h3, h4, label {
    color: #2D3748 !important;
    font-family: 'Segoe UI', Roboto, sans-serif;
}
.block-container p, .block-container .file-status-label,
[data-testid="stSidebar"] p {
    color: #374151 !important;
}

[data-testid="stFileUploader"] {
    background: #F7F4F0 !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] section {
    background: #F7F4F0 !important;
}
[data-testid="stFileUploaderDropzone"],
div[data-testid="stFileUploader"] section {
    background-color: #F7F4F0 !important;
    border: 1px dashed #DDD4CB !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploaderFileName"],
[data-testid="stFileUploaderFile"],
[data-testid="stFileUploader"] li {
    background-color: #E6DDD4 !important;
    border-radius: 8px !important;
    color: #374151 !important;
}

button[kind="primary"] {
    background: linear-gradient(135deg, #6B7280 0%, #5B6573 100%) !important;
    border: 1px solid #5B6573 !important;
    border-radius: 10px !important;
    height: 46px !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important;
    transition: all 0.2s ease !important;
}
button[kind="primary"] *,
button[kind="primary"] p,
button[kind="primary"] span,
button[kind="primary"] div {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 15px !important;
}
button[kind="primary"]:hover {
    background: linear-gradient(135deg, #7A8492 0%, #6B7280 100%) !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.12) !important;
}

button[kind="secondary"] {
    background: #F7F4F0 !important;
    background-color: #F7F4F0 !important;
    border: 1px solid #DDD4CB !important;
    border-radius: 10px !important;
    height: 46px !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
button[kind="secondary"] *,
button[kind="secondary"] p,
button[kind="secondary"] span {
    color: #2D3748 !important;
    -webkit-text-fill-color: #2D3748 !important;
    font-weight: 600 !important;
    font-size: 15px !important;
}
button[kind="secondary"]:hover {
    background: #EAE5DD !important;
    border-color: #CBBFB2 !important;
}

div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] {
    color: #374151 !important;
}
.stRadio [data-baseweb="radio"] svg {
    fill: #6B7280 !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    background: #F7F4F0 !important;
    border: 1px solid #DDD4CB !important;
    border-radius: 10px !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {
    background-color: #F7F4F0 !important;
}

.block-container [data-testid="stVerticalBlockBorderWrapper"] {
    background-color: transparent !important;
    border: none !important;
}

div[data-testid="stSidebar"] div[data-baseweb="slider"] > div {
    background-color: #D6CDC3 !important;
}
div[data-testid="stSidebar"] div[data-baseweb="slider"] > div > div {
    background: #6B7280 !important;
}
div[data-testid="stSidebar"] [role="slider"] {
    background-color: #6B7280 !important;
    border: 2px solid #6B7280 !important;
}

div[data-testid="stSidebar"] div[data-baseweb="tooltip"] {
    background: #6B7280 !important;
    border-radius: 5px !important;
    border: 1px solid #5B6573 !important;
    padding: 2px 6px !important;
}
div[data-testid="stSidebar"] div[data-baseweb="tooltip"] * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 700 !important;
}

div[data-testid="stSidebar"] [data-testid="stSliderTickBar"] > div {
    background: #6B7280 !important;
    border-radius: 4px !important;
    padding: 2px 8px !important;
    margin-top: 8px !important;
    border: 1px solid #5B6573 !important;
}
div[data-testid="stSidebar"] [data-testid="stSliderTickBar"] span {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 700 !important;
}

div[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #374151 !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    margin-bottom: 5px !important;
}

[data-testid="stImage"] img {
    width: 100% !important;
    height: 340px !important;
    object-fit: cover !important;
    border-radius: 10px;
    border: 1px solid #DDD4CB;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: transparent !important;
    border: none !important;
    padding: 0px !important;
}
</style>
""", unsafe_allow_html=True)

with stream.sidebar:
    stream.markdown(
        '<h1 style="font-size: 32px; font-weight: 700; letter-spacing: -0.5px; margin-bottom:0;">'
        '<span style="color: #2D3748;">Face</span>'
        '<span style="color: #6B7280;">Verifier</span></h1>',
        unsafe_allow_html=True
    )
    stream.markdown(
        '<div style="font-size:15px; font-weight:600; color:#7C5C4E; '
        'margin-top:5px; margin-bottom:8px;">Face Similarity Analysis System</div>',
        unsafe_allow_html=True
    )
    stream.markdown(
        '<p style="font-size: 13px; line-height: 1.6; color:#374151;">'
        'Sistem verifikasi wajah untuk membandingkan dua citra menggunakan '
        'InsightFace (ONNX Runtime) dan Principal Component Analysis (PCA).</p>',
        unsafe_allow_html=True
    )

    stream.markdown(
        '<div style="background-color: #F7F4F0; border: 1px solid #DDD4CB; padding: 15px; border-radius: 12px;">'
        '<strong style="color:#7C5C4E;">InsightFace + PCA Engine</strong><br>'
        '<p style="font-size:12px; line-height:1.5; margin:5px 0 0 0; color:#374151;">'
        'Mengekstrak embedding wajah 512-dimensi menggunakan model ONNX '
        '(buffalo_sc), lalu memproyeksikannya ke ruang PCA untuk menghitung '
        'jarak Euclidean dan Cosine Similarity secara efisien.</p></div>',
        unsafe_allow_html=True
    )

    stream.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    stream.sidebar.markdown("### Fitur Tambahan")

    box_fitur = stream.container(border=True)

    with box_fitur:
        stream.markdown(
            '<div style="color:#7C5C4E; font-size:14px; font-weight:700; margin-bottom:12px; '
            'font-family: \'Segoe UI\', Roboto, sans-serif;">PCA Image Compression</div>',
            unsafe_allow_html=True
        )

        n_components_slider = stream.slider(
            "Pilih Komponen Utama (PCA)",
            5,
            150,
            50,
            key="SliderPCA_Core"
        )

stream.markdown(
    '<h2 style="font-size:26px; font-weight:700; color:#2D3748; margin-bottom: 0;">'
    'Verifikasi Komparasi Wajah</h2>',
    unsafe_allow_html=True
)
stream.markdown(
    '<p style="font-size:14px; margin-bottom: 25px; color:#374151;">'
    'Unggah dua foto wajah dalam format JPG atau PNG. Sistem akan menganalisis '
    'koordinat geometris wajah di dalam ruang matriks tereduksi PCA.</p>',
    unsafe_allow_html=True
)

col1, col2 = stream.columns(2)

with col1:
    stream.markdown(
        """<div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
        <span style="background: #F1EAE3; color: #7C5C4E; font-size: 11px; font-weight: 700;
        padding: 4px 10px; border-radius: 6px; border: 1px solid #D6CDC3;">Sampel A</span>
        <span style="font-size: 14px; font-weight: 600; color: #374151;">Foto Masa Kecil</span>
        </div>""",
        unsafe_allow_html=True
    )

    file_baby = stream.file_uploader("Pilih dokumen citra wajah A", type=["jpg", "jpeg", "png"], key="baby", label_visibility="collapsed")

    if file_baby:
        stream.markdown(
            '<p style="font-size:13px; color:#15803D; font-weight:600; margin-top:8px; margin-bottom:8px; text-align:center;">✓ Foto berhasil diunggah</p>',
            unsafe_allow_html=True
        )
        img_a = Image.open(file_baby)
        stream.image(img_a, caption="Pratinjau Foto Masa Kecil", use_column_width=True)

with col2:
    stream.markdown(
        """<div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
        <span style="background: #F1EAE3; color: #7C5C4E; font-size: 11px; font-weight: 700;
        padding: 4px 10px; border-radius: 6px; border: 1px solid #D6CDC3;">Sampel B</span>
        <span style="font-size: 14px; font-weight: 600; color: #374151;">Foto Sekarang</span>
        </div>""",
        unsafe_allow_html=True
    )

    file_adult = stream.file_uploader("Pilih dokumen citra wajah B", type=["jpg", "jpeg", "png"], key="adult", label_visibility="collapsed")

    if file_adult:
        stream.markdown(
            '<p style="font-size:13px; color:#15803D; font-weight:600; margin-top:8px; margin-bottom:8px; text-align:center;">✓ Foto berhasil diunggah</p>',
            unsafe_allow_html=True
        )
        img_b = Image.open(file_adult)
        stream.image(img_b, caption="Pratinjau Foto Sekarang", use_column_width=True)

stream.markdown(
    "<hr style='border: 0; height: 1px; background: #DCD7CE; margin: 25px 0;'>",
    unsafe_allow_html=True
)

btn_col1, btn_col2 = stream.columns([3, 1])
with btn_col1:
    verify_clicked = stream.button("Jalankan Autentikasi", type="primary", use_container_width=True)
with btn_col2:
    reset_clicked = stream.button("Mulai Ulang", type="secondary", use_container_width=True)

if reset_clicked:
    for key in list(stream.session_state.keys()):
        del stream.session_state[key]
    stream.rerun()

if verify_clicked:
    if not file_baby or not file_adult:
        stream.error("⚠️ Gagal Mengevaluasi: Silakan pilih kedua dokumen citra wajah terlebih dahulu.")
    else:
        tmp_dir = tempfile.gettempdir()
        tmp_baby_path = os.path.join(tmp_dir, "tmp_face1.png")
        tmp_adult_path = os.path.join(tmp_dir, "tmp_face2.png")

        try:
            # FIX: konversi ke RGB sebelum disimpan sebagai PNG, agar foto
            # dengan mode RGBA/CMYK/P (alpha channel, palette, dsb.) tidak
            # menghasilkan warna yang salah saat dibaca ulang oleh OpenCV.
            Image.open(file_baby).convert("RGB").save(tmp_baby_path)
            Image.open(file_adult).convert("RGB").save(tmp_adult_path)

            with stream.spinner("Mengekstraksi fitur wajah dan melakukan proyeksi PCA..."):
                stream.session_state["pilihan_res"] = ef.compare_faces(tmp_baby_path, tmp_adult_path)
        except Exception as e:
            stream.error(f"⚠️ Gagal Mengevaluasi: {str(e)}")
        finally:
            if os.path.exists(tmp_baby_path): os.remove(tmp_baby_path)
            if os.path.exists(tmp_adult_path): os.remove(tmp_adult_path)

if "pilihan_res" in stream.session_state:
    res = stream.session_state["pilihan_res"]

    stream.markdown(
        "<hr style='border: 0; height: 1px; background: #DCD7CE; margin: 25px 0;'>",
        unsafe_allow_html=True
    )

    is_match = "Sama" in res["result"]

    if is_match:
        badge_bg, badge_color, badge_border = "#DCFCE7", "#15803D", "#BBF7D0"
        badge_icon = "✓"
    else:
        badge_bg, badge_color, badge_border = "#F3ECE6", "#8B5E3C", "#E3D3C4"
        badge_icon = "✕"

    try:
        val_progress = float(res["similarity"])
    except Exception:
        val_progress = 50.0

    panel_html = f"""
    <div style="background-color: #E6DDD4; padding: 24px; border-radius: 12px; border: 1px solid #D6CDC3; width: 100%; box-sizing: border-box;">
        <h4 style="font-size:16px; font-weight:700; color:#2D3748; margin-top:0; margin-bottom:14px;">Hasil Analisis Biometrik Wajah</h4>
        <div style="margin-bottom:18px;">
            <span style="background:{badge_bg}; color:{badge_color}; font-size:13px; font-weight:700; padding:6px 12px; border-radius:6px; display:inline-block; border:1px solid {badge_border};">
                {badge_icon} {res["result"]}
            </span>
        </div>
        <div style="background-color: #D6CDC3; border-radius: 10px; height: 10px; width: 100%; margin-bottom: 22px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, #7A8B94, #A89F91); height: 100%; width: {val_progress}%;"></div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px;">
            <div style="background: #F7F4F0; padding: 14px; border-radius: 10px; border: 1px solid #DDD4CB;">
                <p style="font-size:11px; color:#6B7280; margin:0 0 4px 0; font-weight:500;">Cosine Similarity</p>
                <strong style="font-size:15px; color:#2D3748;">{res["cosine"]}</strong>
            </div>
            <div style="background: #F7F4F0; padding: 14px; border-radius: 10px; border: 1px solid #DDD4CB;">
                <p style="font-size:11px; color:#6B7280; margin:0 0 4px 0; font-weight:500;">Euclidean Distance</p>
                <strong style="font-size:15px; color:#2D3748;">{res["euclidean"]}</strong>
            </div>
            <div style="background: #F7F4F0; padding: 14px; border-radius: 10px; border: 1px solid #DDD4CB;">
                <p style="font-size:11px; color:#6B7280; margin:0 0 4px 0; font-weight:500;">Ambang Batas (Threshold)</p>
                <strong style="font-size:15px; color:#2D3748;">0.2800</strong>
            </div>
            <div style="background: #F7F4F0; padding: 14px; border-radius: 10px; border: 1px solid #DDD4CB; text-align:center;">
                <p style="font-size:11px; color:#6B7280; margin:0 0 4px 0; font-weight:500;">Kemiripan</p>
                <strong style="font-size:24px; font-weight:700; color:#7C5C4E;">{res["similarity"]}%</strong>
            </div>
        </div>
    </div>
    """
    stream.markdown(panel_html, unsafe_allow_html=True)

stream.markdown(
    "<hr style='border: 0; height: 1px; background: #DCD7CE; margin: 30px 0;'>",
    unsafe_allow_html=True
)
stream.subheader("Simulasi Kompresi Citra dengan PCA")

pilihan_sampel = []
if file_baby:
    pilihan_sampel.append("Sampel A (Foto Masa Kecil)")
if file_adult:
    pilihan_sampel.append("Sampel B (Foto Sekarang)")

if pilihan_sampel:
    pilihan_terpilih = stream.radio("Pilih citra untuk dianalisis:", pilihan_sampel, horizontal=True)

    if stream.button("Proses Kompresi Citra", type="primary"):
        file_target = file_baby if "Sampel A" in pilihan_terpilih else file_adult
        nama_caption = "Sampel A" if "Sampel A" in pilihan_terpilih else "Sampel B"

        img_gray = Image.open(file_target).convert('L')
        img_gray = resize_untuk_pca(img_gray)
        arr_gray = np.array(img_gray)
        res_gray = hitung_pca_core(arr_gray, n_components_slider)

        c_gray1, c_gray2 = stream.columns(2)
        with c_gray1:
            stream.image(img_gray, caption=f"Citra Grayscale Asli ({nama_caption})", use_column_width=True)
        with c_gray2:
            stream.image(res_gray, caption=f"Hasil Rekonstruksi PCA ({n_components_slider} Komponen)", use_column_width=True)