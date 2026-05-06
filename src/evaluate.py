"""
=================================================================
evaluate.py â€” Evaluasi Model CNN
=================================================================
Script ini melakukan:
1. Memuat model terbaik yang sudah di-training
2. Mengevaluasi performa model pada data test
3. Membuat confusion matrix
4. Membuat classification report (precision, recall, f1-score)
5. Menyimpan semua hasil ke folder results/

Jalankan dengan:
  python src/evaluate.py
  
Pastikan sudah menjalankan train.py terlebih dahulu!
=================================================================
"""

import sys
import json
import numpy as np
from pathlib import Path

# Tambahkan folder src ke Python path
sys.path.append(str(Path(__file__).resolve().parent))

from config import (
    BEST_MODEL_PATH, RESULTS_DIR, EMOTION_LABELS
)
from utils import (
    create_data_generators,
    plot_training_history,
    plot_confusion_matrix,
    generate_classification_report
)


def evaluate():
    """
    Fungsi utama untuk mengevaluasi model.
    
    Evaluasi meliputi:
    - Test accuracy & loss
    - Confusion matrix (visualisasi kesalahan prediksi)
    - Classification report (precision, recall, f1-score per kelas)
    """
    
    import tensorflow as tf
    
    print("=" * 60)
    print("  EVALUASI MODEL FACIAL EMOTION RECOGNITION")
    print("=" * 60)
    
    # -------------------------------------------------------
    # STEP 1: CEK MODEL
    # -------------------------------------------------------
    print("\n[1/5] Memeriksa model...")
    
    if not BEST_MODEL_PATH.exists():
        print(f"  ERROR: Model tidak ditemukan di:")
        print(f"  {BEST_MODEL_PATH}")
        print(f"\n  Jalankan training dulu: python src/train.py")
        sys.exit(1)
    
    print(f"  âœ“ Model ditemukan: {BEST_MODEL_PATH}")
    
    # -------------------------------------------------------
    # STEP 2: LOAD MODEL
    # -------------------------------------------------------
    print("\n[2/5] Memuat model...")
    model = tf.keras.models.load_model(str(BEST_MODEL_PATH))
    print("  âœ“ Model berhasil dimuat!")
    
    # -------------------------------------------------------
    # STEP 3: LOAD DATA TEST
    # -------------------------------------------------------
    print("\n[3/5] Memuat data test...")
    _, test_gen = create_data_generators()
    
    # -------------------------------------------------------
    # STEP 4: EVALUASI MODEL
    # -------------------------------------------------------
    print("\n[4/5] Mengevaluasi model pada data test...")
    
    # Hitung loss dan accuracy pada data test
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    
    print(f"\n  Hasil Evaluasi:")
    print(f"  â”œâ”€ Test Loss     : {test_loss:.4f}")
    print(f"  â”œâ”€ Test Accuracy : {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # -------------------------------------------------------
    # STEP 5: PREDIKSI & ANALISIS DETAIL
    # -------------------------------------------------------
    print("\n[5/5] Membuat analisis detail...")
    
    # Prediksi semua data test
    print("  Melakukan prediksi pada seluruh data test...")
    predictions = model.predict(test_gen, verbose=1)
    
    # Ambil label prediksi (index dengan probabilitas tertinggi)
    y_pred = np.argmax(predictions, axis=1)
    
    # Ambil label sebenarnya
    y_true = test_gen.classes
    
    # Buat confusion matrix
    print("\n  Membuat confusion matrix...")
    plot_confusion_matrix(y_true, y_pred)
    
    # Buat classification report
    print("\n  Membuat classification report...")
    generate_classification_report(y_true, y_pred)
    
    # -------------------------------------------------------
    # PLOT TRAINING HISTORY (jika tersedia)
    # -------------------------------------------------------
    history_path = RESULTS_DIR / "training_history.json"
    if history_path.exists():
        print("\n  Membuat ulang grafik training history...")
        with open(str(history_path), 'r') as f:
            history_data = json.load(f)
        
        # Buat objek sederhana yang mirip history Keras
        class SimpleHistory:
            def __init__(self, data):
                self.history = data
        
        plot_training_history(SimpleHistory(history_data))
    
    # -------------------------------------------------------
    # RINGKASAN AKHIR
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("  EVALUASI SELESAI!")
    print("=" * 60)
    print(f"\n  Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"\n  File yang dihasilkan:")
    print(f"  â”œâ”€ {RESULTS_DIR / 'confusion_matrix.png'}")
    print(f"  â”œâ”€ {RESULTS_DIR / 'classification_report.txt'}")
    print(f"  â””â”€ {RESULTS_DIR / 'training_history.png'}")
    
    # Analisis per kelas
    print(f"\n  Analisis per kelas emosi:")
    from sklearn.metrics import accuracy_score
    for i, label in enumerate(EMOTION_LABELS):
        mask = y_true == i
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == i).sum() / mask.sum()
            print(f"  â”œâ”€ {label:>10s}: {class_acc*100:5.1f}% "
                  f"({(y_pred[mask] == i).sum()}/{mask.sum()} benar)")
    
    print(f"\n  Selanjutnya jalankan: python src/realtime_detection.py")
    print()


if __name__ == "__main__":
    evaluate()
