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
MIN_TRAINING_SAMPLES = 2  
DETECTION_SIZE = (320, 320)
MAX_IMAGE_DIM = 1024 

class FaceRecognitionSystem:
    def __init__(self):
        self.pca_handler = None
        self.is_trained = False

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3

        self.app = FaceAnalysis(
            name='buffalo_sc',
            providers=['CPUExecutionProvider'],
            provider_options=[{'session_options': opts}]
        )
        self.app.prepare(ctx_id=0, det_size=DETECTION_SIZE)

    def _resize_if_needed(self, img):
        h, w = img.shape[:2]
        if max(h, w) > MAX_IMAGE_DIM:
            skala = MAX_IMAGE_DIM / max(h, w)
            img = cv2.resize(img, (int(w * skala), int(h * skala)), interpolation=cv2.INTER_AREA)
        return img

    def _extract_buffalo_embedding(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Gambar tidak dapat dibaca. Pastikan file tidak korup.")

        img = self._resize_if_needed(img)
        faces = self.app.get(img)
        
        if len(faces) == 0:
            raise ValueError("Wajah tidak terdeteksi secara jelas. Gunakan foto dengan wajah yang lebih terlihat.")

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

        if len(embeddings_list) == 0:
            print("Peringatan: Tidak ada data training valid. Membuat data sintetis...")
            embeddings_list = [np.random.rand(512) for _ in range(MIN_TRAINING_SAMPLES + 1)]
        elif len(embeddings_list) < MIN_TRAINING_SAMPLES:
            print(f"Peringatan: Kekurangan data training. Menambah data sintetis...")
            jumlah_tambahan = MIN_TRAINING_SAMPLES + 1 - len(embeddings_list)
            embeddings_list.extend(np.random.rand(512) for _ in range(jumlah_tambahan))

        X = np.array(embeddings_list)
        components = min(N_COMPONENTS, X.shape[0], X.shape[1])
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