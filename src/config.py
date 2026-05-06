"""
=================================================================
config.py â€” Konfigurasi Utama Project
=================================================================
File ini berisi semua pengaturan path, parameter training,
dan daftar kelas emosi yang digunakan di seluruh project.

Semua path menggunakan Path dari pathlib agar kompatibel
di Windows maupun Linux/Mac.
=================================================================
"""

import os
from pathlib import Path

# ---------------------------------------------------------------
# BASE DIRECTORY
# ---------------------------------------------------------------
# BASE_DIR menunjuk ke folder utama project (facial-emotion-recognition/)
# Semua path lain diturunkan dari sini agar tetap relatif dan portabel
BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------
# DATASET PATHS
# ---------------------------------------------------------------
# Path ke dataset Arrow asli dari HuggingFace (sebelum dikonversi)
ARROW_DATASET_DIR = BASE_DIR.parent / "fer2013_enhanced"

# Path ke dataset gambar yang sudah dikonversi ke folder per kelas
DATASET_DIR = BASE_DIR / "dataset" / "fer2013"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

# ---------------------------------------------------------------
# HAARCASCADE PATH
# ---------------------------------------------------------------
# File XML untuk deteksi wajah menggunakan Haar Cascade OpenCV
HAARCASCADE_PATH = BASE_DIR / "haarcascade" / "haarcascade_frontalface_default.xml"

# ---------------------------------------------------------------
# MODEL & RESULTS PATHS
# ---------------------------------------------------------------
# Folder untuk menyimpan model yang sudah di-training
MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_emotion_model.keras"

# Folder untuk menyimpan hasil evaluasi (grafik, report, dll)
RESULTS_DIR = BASE_DIR / "results"

# ---------------------------------------------------------------
# IMAGE PARAMETERS
# ---------------------------------------------------------------
# Ukuran gambar input untuk model CNN (48x48 piksel grayscale)
IMG_SIZE = 48
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 1)  # 1 channel karena grayscale

# ---------------------------------------------------------------
# TRAINING PARAMETERS
# ---------------------------------------------------------------
BATCH_SIZE = 64       # Jumlah gambar per batch saat training
EPOCHS = 50           # Jumlah maksimum epoch training
LEARNING_RATE = 0.001 # Learning rate awal untuk optimizer Adam

# ---------------------------------------------------------------
# DAFTAR KELAS EMOSI
# ---------------------------------------------------------------
# 7 kelas emosi pada dataset FER-2013
# PENTING: Urutan ini harus sama dengan urutan label di dataset!
EMOTION_LABELS = [
    "Angry",     # 0
    "Disgust",   # 1
    "Fear",      # 2
    "Happy",     # 3
    "Sad",       # 4
    "Surprise",  # 5
    "Neutral"    # 6
]

NUM_CLASSES = len(EMOTION_LABELS)  # 7 kelas

# ---------------------------------------------------------------
# BUAT FOLDER JIKA BELUM ADA
# ---------------------------------------------------------------
# Otomatis membuat folder yang dibutuhkan saat config di-import
for directory in [MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------
# CETAK KONFIGURASI (untuk debugging)
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  KONFIGURASI PROJECT FACIAL EMOTION RECOGNITION")
    print("=" * 60)
    print(f"  Base Directory     : {BASE_DIR}")
    print(f"  Arrow Dataset      : {ARROW_DATASET_DIR}")
    print(f"  Dataset Directory  : {DATASET_DIR}")
    print(f"  Train Directory    : {TRAIN_DIR}")
    print(f"  Test Directory     : {TEST_DIR}")
    print(f"  Haarcascade Path   : {HAARCASCADE_PATH}")
    print(f"  Models Directory   : {MODELS_DIR}")
    print(f"  Best Model Path    : {BEST_MODEL_PATH}")
    print(f"  Results Directory  : {RESULTS_DIR}")
    print(f"  Image Size         : {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch Size         : {BATCH_SIZE}")
    print(f"  Epochs             : {EPOCHS}")
    print(f"  Learning Rate      : {LEARNING_RATE}")
    print(f"  Emotion Labels     : {EMOTION_LABELS}")
    print(f"  Number of Classes  : {NUM_CLASSES}")
    print("=" * 60)
