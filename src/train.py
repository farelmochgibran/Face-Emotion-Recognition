"""
=================================================================
train.py â€” Training Model CNN untuk Facial Emotion Recognition
=================================================================
Script ini melakukan:
1. Memuat dataset FER-2013 dari folder gambar
2. Membuat model CNN custom
3. Melatih model dengan augmentasi data
4. Menyimpan model terbaik berdasarkan validation accuracy
5. Menyimpan grafik training history

Jalankan dengan:
  python src/train.py

Pastikan sudah menjalankan prepare_dataset.py terlebih dahulu!
=================================================================
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# Tambahkan folder src ke Python path
sys.path.append(str(Path(__file__).resolve().parent))

from config import (
    TRAIN_DIR, TEST_DIR, IMG_SIZE, IMG_SHAPE,
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    MODELS_DIR, BEST_MODEL_PATH, RESULTS_DIR,
    NUM_CLASSES, EMOTION_LABELS
)
from utils import create_data_generators, plot_training_history


def build_cnn_model():
    """
    Membangun arsitektur model CNN custom untuk klasifikasi emosi.
    
    Arsitektur:
    - Input: Gambar grayscale 48x48x1
    - 2 blok konvolusi (Conv2D + BatchNorm + MaxPool + Dropout)
    - 1 blok fully connected (Dense + Dropout)
    - Output: 7 kelas emosi (softmax)
    
    Penjelasan layer:
    - Conv2D: Mendeteksi fitur/pola pada gambar (tepi, tekstur, dll)
    - BatchNormalization: Menstabilkan dan mempercepat training
    - MaxPooling2D: Mengurangi ukuran gambar, mempertahankan fitur penting
    - Dropout: Mencegah overfitting dengan "mematikan" neuron secara acak
    - Dense: Layer fully connected untuk klasifikasi
    - Softmax: Menghasilkan probabilitas untuk setiap kelas
    
    Returns:
        model: Model Keras yang siap di-compile dan di-training
    """
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv2D, BatchNormalization, MaxPooling2D,
        Dropout, Flatten, Dense, Input
    )
    
    model = Sequential(name="EmotionCNN")
    
    # -------------------------------------------------------
    # INPUT LAYER
    # -------------------------------------------------------
    model.add(Input(shape=IMG_SHAPE))  # (48, 48, 1)
    
    # -------------------------------------------------------
    # BLOK KONVOLUSI 1 â€” Mendeteksi fitur dasar (tepi, garis)
    # -------------------------------------------------------
    # Conv2D 64 filter: Mengekstrak 64 fitur berbeda dari gambar
    # Kernel (3,3): Ukuran jendela filter 3x3 piksel
    # padding='same': Output tetap ukuran sama dengan input
    # activation='relu': Fungsi aktivasi (hanya loloskan nilai positif)
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())  # Normalisasi output setiap batch
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Ukuran jadi 24x24
    model.add(Dropout(0.25))  # Matikan 25% neuron secara acak
    
    # -------------------------------------------------------
    # BLOK KONVOLUSI 2 â€” Mendeteksi fitur kompleks (bentuk, pola)
    # -------------------------------------------------------
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Ukuran jadi 12x12
    model.add(Dropout(0.25))
    
    # -------------------------------------------------------
    # BLOK KONVOLUSI 3 â€” Mendeteksi fitur level tinggi (ekspresi)
    # -------------------------------------------------------
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Ukuran jadi 6x6
    model.add(Dropout(0.25))
    
    # -------------------------------------------------------
    # FLATTEN â€” Ubah dari 2D ke 1D
    # -------------------------------------------------------
    # Mengubah output konvolusi (6x6x256) menjadi vektor 1D
    model.add(Flatten())
    
    # -------------------------------------------------------
    # FULLY CONNECTED â€” Klasifikasi berdasarkan fitur
    # -------------------------------------------------------
    model.add(Dense(1024, activation='relu'))  # 1024 neuron
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # Matikan 50% neuron (agresif, cegah overfitting)
    
    # -------------------------------------------------------
    # OUTPUT LAYER â€” 7 kelas emosi
    # -------------------------------------------------------
    # Softmax menghasilkan probabilitas untuk setiap kelas
    # Total probabilitas = 1.0
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    return model


def train():
    """
    Fungsi utama untuk melatih model CNN.
    
    Proses:
    1. Memuat data training dan testing
    2. Membangun model CNN
    3. Compile model (tentukan optimizer, loss, metric)
    4. Set callback (checkpoint, early stopping, reduce LR)
    5. Training model
    6. Simpan grafik training history
    """
    
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import (
        ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    )
    
    print("=" * 60)
    print("  TRAINING MODEL FACIAL EMOTION RECOGNITION")
    print("=" * 60)
    
    # -------------------------------------------------------
    # CEK DATASET
    # -------------------------------------------------------
    if not TRAIN_DIR.exists() or not any(TRAIN_DIR.iterdir()):
        print("\n  ERROR: Dataset belum disiapkan!")
        print("  Jalankan dulu: python src/prepare_dataset.py")
        sys.exit(1)
    
    # -------------------------------------------------------
    # STEP 1: MEMUAT DATA
    # -------------------------------------------------------
    print("\n[1/5] Memuat dataset...")
    train_gen, test_gen = create_data_generators()
    
    # -------------------------------------------------------
    # STEP 2: MEMBANGUN MODEL
    # -------------------------------------------------------
    print("\n[2/5] Membangun arsitektur model CNN...")
    model = build_cnn_model()
    
    # Tampilkan ringkasan model
    print("\n  Arsitektur Model:")
    print("  " + "-" * 56)
    model.summary()
    
    # -------------------------------------------------------
    # STEP 3: COMPILE MODEL
    # -------------------------------------------------------
    print("\n[3/5] Compile model...")
    
    # Adam: Optimizer adaptif yang populer dan efektif
    # categorical_crossentropy: Loss function untuk multi-class
    # accuracy: Metrik yang kita pantau
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("  âœ“ Optimizer : Adam (lr={})".format(LEARNING_RATE))
    print("  âœ“ Loss      : Categorical Crossentropy")
    print("  âœ“ Metric    : Accuracy")
    
    # -------------------------------------------------------
    # STEP 4: SETUP CALLBACKS
    # -------------------------------------------------------
    print("\n[4/5] Menyiapkan callbacks...")
    
    callbacks = [
        # ModelCheckpoint: Simpan model terbaik berdasarkan val_accuracy
        # - monitor='val_accuracy': Pantau akurasi validasi
        # - save_best_only=True: Hanya simpan jika lebih baik dari sebelumnya
        # - mode='max': Simpan yang val_accuracy paling tinggi
        ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # EarlyStopping: Hentikan training jika tidak ada peningkatan
        # - patience=10: Tunggu 10 epoch tanpa peningkatan sebelum berhenti
        # - restore_best_weights: Kembalikan bobot terbaik saat berhenti
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # ReduceLROnPlateau: Kurangi learning rate jika loss stagnant
        # - factor=0.5: Kurangi LR menjadi setengahnya
        # - patience=5: Tunggu 5 epoch tanpa peningkatan
        # - min_lr=1e-7: Batas minimum learning rate
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print("  âœ“ ModelCheckpoint  (simpan model terbaik)")
    print("  âœ“ EarlyStopping    (berhenti jika tidak ada peningkatan)")
    print("  âœ“ ReduceLROnPlateau (kurangi learning rate otomatis)")
    
    # -------------------------------------------------------
    # STEP 5: MULAI TRAINING!
    # -------------------------------------------------------
    print("\n[5/5] Mulai training...")
    print(f"  Epoch  : {EPOCHS}")
    print(f"  Batch  : {BATCH_SIZE}")
    print(f"  Train  : {train_gen.samples} gambar")
    print(f"  Test   : {test_gen.samples} gambar")
    print("-" * 60)
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=test_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # -------------------------------------------------------
    # SIMPAN HASIL
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("  TRAINING SELESAI!")
    print("=" * 60)
    
    # Simpan grafik training
    print("\n  Menyimpan grafik training history...")
    plot_training_history(history)
    
    # Simpan training history ke JSON (untuk evaluate.py)
    history_path = RESULTS_DIR / "training_history.json"
    history_data = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
    }
    with open(str(history_path), 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"  âœ“ Training history disimpan: {history_path}")
    
    # Tampilkan ringkasan akhir
    best_train_acc = max(history.history['accuracy'])
    best_val_acc = max(history.history['val_accuracy'])
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"\n  Ringkasan:")
    print(f"  â”œâ”€ Best Training Accuracy   : {best_train_acc:.4f} ({best_train_acc*100:.2f}%)")
    print(f"  â”œâ”€ Best Validation Accuracy : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  â”œâ”€ Final Training Loss      : {final_train_loss:.4f}")
    print(f"  â”œâ”€ Final Validation Loss    : {final_val_loss:.4f}")
    print(f"  â”œâ”€ Model tersimpan di       : {BEST_MODEL_PATH}")
    print(f"  â””â”€ Grafik tersimpan di      : {RESULTS_DIR}")
    
    print(f"\n  Selanjutnya jalankan: python src/evaluate.py")
    print()


if __name__ == "__main__":
    train()
