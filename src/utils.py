"""
=================================================================
utils.py â€” Fungsi Bantuan (Utility Functions)
=================================================================
File ini berisi fungsi-fungsi pembantu yang digunakan oleh
train.py dan evaluate.py, seperti:
- Membuat data generator untuk training dan testing
- Plotting grafik accuracy dan loss
- Membuat confusion matrix
- Membuat classification report
=================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interaktif agar bisa save tanpa GUI
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    ConfusionMatrixDisplay
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import (
    TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE, 
    EMOTION_LABELS, RESULTS_DIR
)


def create_data_generators():
    """
    Membuat ImageDataGenerator untuk training dan testing.
    
    Training data menggunakan augmentasi (rotasi, flip, zoom, dll)
    untuk memperbanyak variasi data dan mengurangi overfitting.
    
    Test data TIDAK di-augmentasi, hanya dinormalisasi.
    
    Returns:
        train_generator: Generator data training dengan augmentasi
        test_generator: Generator data testing tanpa augmentasi
    """
    
    # -------------------------------------------------------
    # AUGMENTASI DATA TRAINING
    # -------------------------------------------------------
    # Augmentasi membantu model belajar lebih baik dengan
    # membuat variasi dari gambar yang sudah ada:
    # - rescale: normalisasi piksel dari 0-255 ke 0-1
    # - rotation_range: rotasi acak hingga 15 derajat
    # - width/height_shift: geser horizontal/vertikal hingga 15%
    # - shear_range: distorsi shear hingga 15%
    # - zoom_range: zoom in/out hingga 15%
    # - horizontal_flip: flip horizontal (cermin)
    # - fill_mode: isi piksel kosong dengan piksel terdekat
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,         # Normalisasi ke 0-1
        rotation_range=15,            # Rotasi acak Â±15Â°
        width_shift_range=0.15,       # Geser horizontal Â±15%
        height_shift_range=0.15,      # Geser vertikal Â±15%
        shear_range=0.15,             # Distorsi shear Â±15%
        zoom_range=0.15,              # Zoom Â±15%
        horizontal_flip=True,         # Flip horizontal
        fill_mode='nearest'           # Isi piksel kosong
    )
    
    # -------------------------------------------------------
    # DATA TESTING (TANPA AUGMENTASI)
    # -------------------------------------------------------
    # Data test hanya dinormalisasi, TIDAK di-augmentasi
    # karena kita ingin mengevaluasi pada data asli
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0           # Normalisasi ke 0-1 saja
    )
    
    # -------------------------------------------------------
    # BUAT GENERATOR DARI FOLDER
    # -------------------------------------------------------
    # flow_from_directory membaca gambar dari folder dan
    # otomatis memberi label berdasarkan nama subfolder
    
    print(f"  Memuat data training dari: {TRAIN_DIR}")
    train_generator = train_datagen.flow_from_directory(
        directory=str(TRAIN_DIR),
        target_size=(IMG_SIZE, IMG_SIZE),  # Resize ke 48x48
        color_mode='grayscale',            # Grayscale (1 channel)
        class_mode='categorical',          # One-hot encoding
        batch_size=BATCH_SIZE,
        shuffle=True                       # Acak urutan data
    )
    
    print(f"  Memuat data testing dari: {TEST_DIR}")
    test_generator = test_datagen.flow_from_directory(
        directory=str(TEST_DIR),
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=False                      # JANGAN acak untuk evaluasi
    )
    
    # Tampilkan info kelas yang ditemukan
    print(f"\n  Kelas yang ditemukan: {train_generator.class_indices}")
    print(f"  Jumlah data training: {train_generator.samples}")
    print(f"  Jumlah data testing : {test_generator.samples}")
    
    return train_generator, test_generator


def plot_training_history(history, save_path=None):
    """
    Membuat grafik accuracy dan loss selama training.
    
    Grafik ini membantu kita melihat:
    - Apakah model belajar dengan baik (accuracy naik)
    - Apakah terjadi overfitting (training bagus, validation jelek)
    - Kapan model mulai converge (stabil)
    
    Args:
        history: Object history dari model.fit()
        save_path: Path untuk menyimpan grafik (opsional)
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # -------------------------------------------------------
    # GRAFIK ACCURACY
    # -------------------------------------------------------
    axes[0].plot(history.history['accuracy'], 
                 label='Training Accuracy', linewidth=2, color='#2196F3')
    axes[0].plot(history.history['val_accuracy'], 
                 label='Validation Accuracy', linewidth=2, color='#FF5722')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # -------------------------------------------------------
    # GRAFIK LOSS
    # -------------------------------------------------------
    axes[1].plot(history.history['loss'], 
                 label='Training Loss', linewidth=2, color='#2196F3')
    axes[1].plot(history.history['val_loss'], 
                 label='Validation Loss', linewidth=2, color='#FF5722')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Simpan grafik
    if save_path is None:
        save_path = RESULTS_DIR / "training_history.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  âœ“ Grafik training disimpan: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Membuat dan menyimpan confusion matrix.
    
    Confusion matrix menunjukkan:
    - Diagonal: prediksi yang benar
    - Off-diagonal: prediksi yang salah
    - Baris: label sebenarnya
    - Kolom: label prediksi
    
    Args:
        y_true: Label sebenarnya (array of int)
        y_pred: Label prediksi (array of int)
        save_path: Path untuk menyimpan grafik (opsional)
    """
    
    # Hitung confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Buat visualisasi
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=EMOTION_LABELS
    )
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Simpan
    if save_path is None:
        save_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  âœ“ Confusion matrix disimpan: {save_path}")


def generate_classification_report(y_true, y_pred, save_path=None):
    """
    Membuat classification report (precision, recall, f1-score).
    
    Penjelasan metrik:
    - Precision: Dari semua yang diprediksi kelas X, berapa yang benar?
    - Recall: Dari semua yang memang kelas X, berapa yang terdeteksi?
    - F1-Score: Rata-rata harmonis precision dan recall
    - Support: Jumlah sampel untuk kelas tersebut
    
    Args:
        y_true: Label sebenarnya (array of int)
        y_pred: Label prediksi (array of int)
        save_path: Path untuk menyimpan report (opsional)
    """
    
    # Buat report
    report = classification_report(
        y_true, y_pred,
        target_names=EMOTION_LABELS,
        digits=4  # 4 digit desimal
    )
    
    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)
    
    # Simpan ke file teks
    if save_path is None:
        save_path = RESULTS_DIR / "classification_report.txt"
    with open(str(save_path), 'w') as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(report)
    print(f"  âœ“ Classification report disimpan: {save_path}")
    
    return report
