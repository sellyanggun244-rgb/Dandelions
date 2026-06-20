import os
import numpy as np
import pickle
import cv2
import streamlit as st
from sklearn.decomposition import PCA
from insightface.app import FaceAnalysis
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "trained_pca_model.pkl")
N_COMPONENTS = 20
MIN_TRAINING_SAMPLES = 2  # minimal sampel valid agar PCA bermakna

# OPTIMASI MEMORY: det_size menentukan resolusi internal yang dipakai
# detector wajah InsightFace. Makin besar, makin akurat untuk wajah kecil
# di foto crowd/CCTV, tapi makin berat RAM & CPU-nya. Untuk foto wajah
# tunggal seperti use case app ini, 320x320 sudah lebih dari cukup dan
# jauh lebih ringan dibanding 640x640 — penting di Streamlit Cloud free
# tier yang RAM-nya dibatasi 1GB.
DETECTION_SIZE = (320, 320)

class FaceRecognitionSystem:
    def __init__(self):
        self.pca_handler = None
        self.is_trained = False

        # OPTIMASI: Membatasi penggunaan thread ONNX Runtime agar server tidak overload
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3

        # Inisialisasi InsightFace dengan menyertakan konfigurasi session_options
        self.app = FaceAnalysis(
            name='buffalo_sc',
            providers=['CPUExecutionProvider'],
            provider_options=[{'session_options': opts}]
        )

        self.app.prepare(ctx_id=0, det_size=DETECTION_SIZE)

    # OPTIMASI MEMORY: batas dimensi maksimum gambar yang diproses. Foto dari
    # HP modern bisa 4000x3000px atau lebih — men-decode dan mendeteksi wajah
    # pada resolusi penuh itu mahal di RAM/CPU tanpa menambah akurasi berarti,
    # karena detector sudah punya det_size internal sendiri.
    MAX_IMAGE_DIM = 1024

    def _resize_if_needed(self, img):
        h, w = img.shape[:2]
        if max(h, w) > self.MAX_IMAGE_DIM:
            skala = self.MAX_IMAGE_DIM / max(h, w)
            img = cv2.resize(img, (int(w * skala), int(h * skala)), interpolation=cv2.INTER_AREA)
        return img

    def _extract_buffalo_embedding(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Gambar tidak dapat dibaca. Pastikan file tidak korup dan formatnya didukung (JPG/PNG).")

        img = self._resize_if_needed(img)

        faces = self.app.get(img)
        if len(faces) == 0:
            raise ValueError("Wajah tidak terdeteksi secara jelas. Gunakan foto dengan wajah yang lebih terlihat.")

        # FIX: jika terdeteksi lebih dari satu wajah, ambil wajah dengan area
        # bounding box terbesar (asumsi: wajah utama/terdekat ke kamera),
        # bukan asal indeks pertama dari hasil deteksi.
        if len(faces) > 1:
            faces = sorted(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True
            )

        embedding = faces[0].normed_embedding
        return embedding

    def train_with_deepface(self, image_paths):
        embeddings_list = []
        print("Memulai ekstraksi fitur dataset training via Buffalo...")

        for path in image_paths:
            try:
                emb = self._extract_buffalo_embedding(path)
                embeddings_list.append(emb)
            except Exception as e:
                print(f"Melewati '{path}': {e}")
                continue

        # FIX: sebelumnya, jika embeddings_list < 2 maka SEMUA embedding asli
        # (termasuk yang sudah berhasil diekstrak) dibuang dan diganti data
        # acak sepenuhnya, membuat model PCA tidak representatif sama sekali.
        # Sekarang data asli yang valid tetap dipakai sebagai dasar, dan hanya
        # ditambah sampel sintetis secukupnya supaya jumlah total mencukupi
        # MIN_TRAINING_SAMPLES untuk PCA bisa di-fit.
        if len(embeddings_list) == 0:
            print("Peringatan: Tidak ada data training valid sama sekali. Membuat data sintetis sementara...")
            embeddings_list = [np.random.rand(512) for _ in range(MIN_TRAINING_SAMPLES + 1)]
        elif len(embeddings_list) < MIN_TRAINING_SAMPLES:
            print(f"Peringatan: Hanya {len(embeddings_list)} data training valid. Menambah data sintetis pelengkap...")
            jumlah_tambahan = MIN_TRAINING_SAMPLES + 1 - len(embeddings_list)
            embeddings_list.extend(np.random.rand(512) for _ in range(jumlah_tambahan))

        X = np.array(embeddings_list)

        components = min(N_COMPONENTS, X.shape[0], X.shape[1])
        # FIX: PCA butuh n_components >= 1, jaga-jaga jika suatu saat X kosong/aneh
        components = max(components, 1)

        self.pca_handler = PCA(n_components=components)
        self.pca_handler.fit(X)
        self.is_trained = True
        self.save_model()
        print("Proses training PCA sukses!")

    def save_model(self):
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump({'pca_handler': self.pca_handler}, f)

    def load_model(self):
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.pca_handler = data['pca_handler']
                    self.is_trained = True
                return True
            except Exception as e:
                print(f"Gagal memuat model PCA: {e}")
                return False
        return False

@st.cache_resource(show_spinner="Memuat model pengenalan wajah (hanya sekali per sesi server)...")
def get_model():
    """
    OPTIMASI MEMORY: @st.cache_resource memastikan FaceRecognitionSystem
    (yang memuat model ONNX InsightFace ke memory) hanya dibuat SEKALI
    selama container Streamlit hidup, bukan setiap kali script di-rerun
    (yang terjadi tiap klik tombol/interaksi). Tanpa ini, re-inisialisasi
    berulang adalah penyebab paling umum app kena "resource limit
    exceeded" di Streamlit Community Cloud.
    """
    instance = FaceRecognitionSystem()
    instance.load_model()
    return instance

model = get_model()

def run_training_manually():
    TRAINING_FOLDER = os.path.join(BASE_DIR, "..", "static", "training")
    if not os.path.exists(TRAINING_FOLDER):
        TRAINING_FOLDER = os.path.join(BASE_DIR, "training")

    training_images = []
    if os.path.exists(TRAINING_FOLDER):
        for root, dirs, files in os.walk(TRAINING_FOLDER):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    training_images.append(os.path.join(root, file))

    model.train_with_deepface(training_images)

def compare_faces(image1_path, image2_path):
    if not model.is_trained:
        run_training_manually()

    try:
        emb1 = model._extract_buffalo_embedding(image1_path)
        emb2 = model._extract_buffalo_embedding(image2_path)

        vec1 = emb1.reshape(1, -1)
        vec2 = emb2.reshape(1, -1)

        if model.pca_handler is not None:
            proj1 = model.pca_handler.transform(vec1).flatten()
            proj2 = model.pca_handler.transform(vec2).flatten()
        else:
            proj1 = vec1.flatten()
            proj2 = vec2.flatten()

        dot_product = np.dot(proj1, proj2)
        norm1 = np.linalg.norm(proj1)
        norm2 = np.linalg.norm(proj2)

        cosine_sim = float(dot_product / (norm1 * norm2)) if (norm1 != 0 and norm2 != 0) else 0.0
        euclidean_dist = float(np.linalg.norm(proj1 - proj2))

        # ---- LOGIKA EVALUASI DINAMIS BERBASIS ANGULAR MARGIN ----
        age_threshold = 0.28

        if cosine_sim >= age_threshold:
            kesimpulan = "Kemungkinan Orang Yang Sama"
            raw_score = 70.0 + ((cosine_sim - age_threshold) / (1.0 - age_threshold)) * 30.0
            similarity_percentage = round(min(max(raw_score, 70.0), 98.0), 2)
        else:
            kesimpulan = "Kemungkinan Orang Berbeda"
            raw_score = (cosine_sim / age_threshold) * 60.0
            similarity_percentage = round(min(max(raw_score, 10.0), 65.0), 2)

        return {
            "similarity": similarity_percentage,
            "cosine": round(cosine_sim, 4),
            "euclidean": round(euclidean_dist, 4),
            "result": kesimpulan
        }

    except Exception as e:
        print(f"Gagal memproses kecocokan wajah: {str(e)}")
        raise ValueError(str(e))