"""
=================================================================
realtime_detection.py â€” Deteksi Emosi Real-Time via Webcam
=================================================================
Script ini melakukan:
1. Memuat model CNN yang sudah di-training
2. Memuat Haar Cascade untuk deteksi wajah
3. Membuka webcam
4. Mendeteksi wajah pada setiap frame
5. Memprediksi emosi pada setiap wajah yang terdeteksi
6. Menampilkan bounding box, label emosi, dan confidence

Kontrol:
  - Tekan 'q' untuk keluar
  - Tekan 's' untuk screenshot

Jalankan dengan:
  python src/realtime_detection.py
=================================================================
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Tambahkan folder src ke Python path
sys.path.append(str(Path(__file__).resolve().parent))

from config import (
    BEST_MODEL_PATH, HAARCASCADE_PATH, 
    IMG_SIZE, EMOTION_LABELS, RESULTS_DIR
)


def run_realtime_detection():
    """
    Menjalankan deteksi emosi wajah secara real-time menggunakan webcam.
    
    Alur kerja setiap frame:
    1. Baca frame dari webcam
    2. Konversi ke grayscale
    3. Deteksi wajah menggunakan Haar Cascade
    4. Untuk setiap wajah:
       a. Crop area wajah
       b. Resize ke 48x48 piksel
       c. Normalisasi piksel ke 0-1
       d. Prediksi emosi menggunakan model CNN
       e. Gambar bounding box dan label
    5. Tampilkan frame
    """
    
    import tensorflow as tf
    
    print("=" * 60)
    print("  REAL-TIME EMOTION DETECTION")
    print("=" * 60)
    
    # -------------------------------------------------------
    # STEP 1: LOAD MODEL
    # -------------------------------------------------------
    print("\n[1/3] Memuat model CNN...")
    
    if not BEST_MODEL_PATH.exists():
        print(f"  ERROR: Model tidak ditemukan di: {BEST_MODEL_PATH}")
        print(f"  Jalankan training dulu: python src/train.py")
        sys.exit(1)
    
    model = tf.keras.models.load_model(str(BEST_MODEL_PATH))
    print(f"  âœ“ Model berhasil dimuat!")
    
    # -------------------------------------------------------
    # STEP 2: LOAD HAAR CASCADE
    # -------------------------------------------------------
    print("\n[2/3] Memuat Haar Cascade...")
    
    if not HAARCASCADE_PATH.exists():
        print(f"  ERROR: Haar Cascade tidak ditemukan di: {HAARCASCADE_PATH}")
        print(f"  Pastikan file 'haarcascade_frontalface_default.xml' ada di folder haarcascade/")
        sys.exit(1)
    
    face_cascade = cv2.CascadeClassifier(str(HAARCASCADE_PATH))
    
    if face_cascade.empty():
        print("  ERROR: Gagal memuat Haar Cascade!")
        sys.exit(1)
    
    print(f"  âœ“ Haar Cascade berhasil dimuat!")
    
    # -------------------------------------------------------
    # STEP 3: BUKA WEBCAM
    # -------------------------------------------------------
    print("\n[3/3] Membuka webcam...")
    
    # cv2.VideoCapture(0) = webcam default
    # Ganti 0 dengan 1, 2, dll jika punya banyak kamera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("  ERROR: Tidak bisa membuka webcam!")
        print("  Pastikan webcam tersambung dan tidak digunakan aplikasi lain.")
        sys.exit(1)
    
    # Set resolusi webcam (opsional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("  âœ“ Webcam berhasil dibuka!")
    print("\n  Kontrol:")
    print("  â”œâ”€ Tekan 'q' untuk keluar")
    print("  â””â”€ Tekan 's' untuk screenshot")
    print("\n  Menjalankan deteksi real-time...\n")
    
    # -------------------------------------------------------
    # WARNA UNTUK SETIAP EMOSI (BGR format)
    # -------------------------------------------------------
    # Setiap emosi punya warna unik agar mudah dibedakan
    emotion_colors = {
        "Angry":    (0, 0, 255),     # Merah
        "Disgust":  (0, 128, 0),     # Hijau tua
        "Fear":     (128, 0, 128),   # Ungu
        "Happy":    (0, 255, 255),   # Kuning
        "Sad":      (255, 0, 0),     # Biru
        "Surprise": (0, 165, 255),   # Oranye
        "Neutral":  (128, 128, 128)  # Abu-abu
    }
    
    # -------------------------------------------------------
    # LOOP UTAMA
    # -------------------------------------------------------
    frame_count = 0
    
    while True:
        # Baca frame dari webcam
        ret, frame = cap.read()
        
        if not ret:
            print("  âš  Gagal membaca frame dari webcam")
            break
        
        frame_count += 1
        
        # Konversi frame ke grayscale untuk deteksi wajah
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Deteksi wajah menggunakan Haar Cascade
        # - scaleFactor=1.3: Faktor skala untuk multi-scale detection
        # - minNeighbors=5: Minimum tetangga untuk mengurangi false positive
        # - minSize=(48,48): Ukuran minimum wajah yang dideteksi
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.3, 
            minNeighbors=5,
            minSize=(48, 48),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Proses setiap wajah yang terdeteksi
        for (x, y, w, h) in faces:
            # Crop area wajah dari gambar grayscale
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize ke 48x48 piksel (sesuai input model)
            face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
            
            # Normalisasi piksel ke range 0-1
            face_normalized = face_resized.astype('float32') / 255.0
            
            # Reshape ke format yang diharapkan model: (1, 48, 48, 1)
            # - 1: batch size (1 gambar)
            # - 48x48: ukuran gambar
            # - 1: channel (grayscale)
            face_input = face_normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
            
            # Prediksi emosi menggunakan model
            prediction = model.predict(face_input, verbose=0)
            
            # Ambil kelas dengan probabilitas tertinggi
            emotion_idx = np.argmax(prediction[0])
            emotion_label = EMOTION_LABELS[emotion_idx]
            confidence = prediction[0][emotion_idx]
            
            # Ambil warna untuk emosi ini
            color = emotion_colors.get(emotion_label, (255, 255, 255))
            
            # Gambar bounding box di sekitar wajah
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Background untuk teks label
            label_text = f"{emotion_label}: {confidence*100:.1f}%"
            text_size = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )[0]
            
            # Gambar background rectangle untuk teks
            cv2.rectangle(
                frame, 
                (x, y - text_size[1] - 10), 
                (x + text_size[0] + 5, y), 
                color, -1  # -1 = filled rectangle
            )
            
            # Tulis label emosi dan confidence
            cv2.putText(
                frame, label_text,
                (x + 2, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,                       # Ukuran font
                (255, 255, 255),           # Warna teks (putih)
                2                          # Ketebalan
            )
            
            # Tampilkan confidence bar di bawah wajah (opsional)
            bar_width = int(w * confidence)
            cv2.rectangle(frame, (x, y+h+5), (x+bar_width, y+h+15), color, -1)
            cv2.rectangle(frame, (x, y+h+5), (x+w, y+h+15), color, 1)
        
        # -------------------------------------------------------
        # INFORMASI PADA FRAME
        # -------------------------------------------------------
        # Jumlah wajah terdeteksi
        info_text = f"Wajah terdeteksi: {len(faces)}"
        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Instruksi
        cv2.putText(frame, "Tekan 'q' untuk keluar | 's' untuk screenshot",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Tampilkan frame
        cv2.imshow("Facial Emotion Recognition", frame)
        
        # -------------------------------------------------------
        # KEYBOARD INPUT
        # -------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        
        # Tekan 'q' untuk keluar
        if key == ord('q'):
            print("\n  âœ“ Keluar dari deteksi real-time.")
            break
        
        # Tekan 's' untuk screenshot
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = RESULTS_DIR / f"screenshot_{timestamp}.png"
            cv2.imwrite(str(screenshot_path), frame)
            print(f"  âœ“ Screenshot disimpan: {screenshot_path}")
    
    # Bersihkan resource
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n  Terima kasih telah menggunakan Facial Emotion Recognition!")
    print()


if __name__ == "__main__":
    run_realtime_detection()
