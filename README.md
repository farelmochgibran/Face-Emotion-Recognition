# 🎭 Facial Emotion Recognition

Sistem deteksi emosi wajah berbasis **Convolutional Neural Network (CNN)** menggunakan dataset **FER-2013**. Project ini dapat mengenali 7 emosi dasar manusia secara real-time melalui webcam.

---

## 📋 Deskripsi

Project ini membangun model deep learning untuk mengenali ekspresi wajah manusia dari gambar grayscale berukuran 48x48 piksel. Model dilatih menggunakan dataset FER-2013 yang berisi sekitar 35.000+ gambar wajah.

### 7 Emosi yang Dikenali:
| No | Emosi | Deskripsi |
|----|-------|-----------|
| 0 | 😠 Angry | Marah |
| 1 | 🤢 Disgust | Jijik |
| 2 | 😨 Fear | Takut |
| 3 | 😊 Happy | Senang |
| 4 | 😢 Sad | Sedih |
| 5 | 😲 Surprise | Terkejut |
| 6 | 😐 Neutral | Netral |

### Target Akurasi
Dataset FER-2013 adalah benchmark yang **cukup menantang**. Akurasi realistis untuk model CNN custom:
- **Baseline**: ~50-55%
- **Target**: ~60-68%
- **State-of-the-art**: ~73-76% (dengan arsitektur kompleks)

> ⚠️ Kelas **Disgust** memiliki data paling sedikit, sehingga biasanya akurasinya paling rendah.

---

## 📁 Struktur Folder

```
facial-emotion-recognition/
│
├── dataset/
│   └── fer2013/
│       ├── train/              # Data training (~25.000 gambar)
│       │   ├── angry/
│       │   ├── disgust/
│       │   ├── fear/
│       │   ├── happy/
│       │   ├── neutral/
│       │   ├── sad/
│       │   └── surprise/
│       │
│       └── test/               # Data testing (~10.000 gambar)
│           ├── angry/
│           ├── disgust/
│           ├── fear/
│           ├── happy/
│           ├── neutral/
│           ├── sad/
│           └── surprise/
│
├── haarcascade/
│   └── haarcascade_frontalface_default.xml
│
├── models/                     # Model hasil training
│   └── best_emotion_model.keras
│
├── results/                    # Hasil evaluasi
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── training_history.json
│
├── src/
│   ├── config.py               # Konfigurasi utama
│   ├── prepare_dataset.py      # Konversi dataset Arrow → gambar
│   ├── train.py                # Training model CNN
│   ├── evaluate.py             # Evaluasi model
│   ├── realtime_detection.py   # Deteksi real-time via webcam
│   └── utils.py                # Fungsi bantuan
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Instalasi

### Prasyarat
- Python 3.10 atau lebih baru
- Webcam (untuk deteksi real-time)
- GPU (opsional, tapi sangat disarankan untuk training)

### Langkah Instalasi

#### 1. Buka terminal di folder project

```bash
cd "d:\Aku adalah Project\facial-emotion-recognition"
```

#### 2. Buat virtual environment

```bash
python -m venv venv
```

#### 3. Aktifkan virtual environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

> 💡 Jika ada error "execution policy", jalankan di PowerShell sebagai Admin:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

#### 4. Install library

```bash
pip install -r requirements.txt
```

---

## 🗂️ Persiapan Dataset

### Dataset FER-2013

Dataset menggunakan format HuggingFace Arrow yang sudah ada di folder `fer2013_enhanced/`. Script `prepare_dataset.py` akan mengkonversi format ini menjadi folder gambar per kelas.

**Jalankan konversi dataset:**

```bash
python src/prepare_dataset.py
```

Script ini akan:
1. Membaca file Arrow dari `fer2013_enhanced/`
2. Mengkonversi setiap entry menjadi file gambar PNG
3. Menyimpan ke folder `dataset/fer2013/train/` dan `dataset/fer2013/test/`
4. Menggabungkan split validation ke test

### Haar Cascade

File `haarcascade_frontalface_default.xml` sudah disiapkan di folder `haarcascade/`. File ini digunakan OpenCV untuk mendeteksi wajah pada frame webcam.

---

## 🚀 Cara Menjalankan

### Urutan Langkah (Step by Step)

```
┌─────────────────────────┐
│  1. Prepare Dataset     │  python src/prepare_dataset.py
│  (Konversi Arrow→Gambar)│
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  2. Training Model      │  python src/train.py
│  (Latih CNN)            │
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  3. Evaluasi Model      │  python src/evaluate.py
│  (Cek performa)         │
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  4. Real-Time Detection │  python src/realtime_detection.py
│  (Webcam)               │
└─────────────────────────┘
```

### Step 1: Konversi Dataset
```bash
python src/prepare_dataset.py
```
Waktu: ~2-5 menit (tergantung performa disk)

### Step 2: Training Model
```bash
python src/train.py
```
Waktu: ~30-60 menit (CPU) atau ~5-15 menit (GPU)

### Step 3: Evaluasi Model
```bash
python src/evaluate.py
```
Waktu: ~1-2 menit

### Step 4: Deteksi Real-Time
```bash
python src/realtime_detection.py
```
- Tekan **'q'** untuk keluar
- Tekan **'s'** untuk screenshot

---

## 🏗️ Arsitektur Model CNN

```
Input (48x48x1 Grayscale)
        │
        ▼
┌─── BLOK KONVOLUSI 1 ───┐
│  Conv2D(64, 3x3)       │
│  BatchNormalization     │
│  Conv2D(64, 3x3)       │
│  BatchNormalization     │
│  MaxPooling2D(2x2)     │ → Output: 24x24x64
│  Dropout(0.25)         │
└─────────┬──────────────┘
          ▼
┌─── BLOK KONVOLUSI 2 ───┐
│  Conv2D(128, 3x3)      │
│  BatchNormalization     │
│  Conv2D(128, 3x3)      │
│  BatchNormalization     │
│  MaxPooling2D(2x2)     │ → Output: 12x12x128
│  Dropout(0.25)         │
└─────────┬──────────────┘
          ▼
┌─── BLOK KONVOLUSI 3 ───┐
│  Conv2D(256, 3x3)      │
│  BatchNormalization     │
│  Conv2D(256, 3x3)      │
│  BatchNormalization     │
│  MaxPooling2D(2x2)     │ → Output: 6x6x256
│  Dropout(0.25)         │
└─────────┬──────────────┘
          ▼
┌─── FULLY CONNECTED ────┐
│  Flatten               │
│  Dense(1024, ReLU)     │
│  BatchNormalization     │
│  Dropout(0.5)          │
│  Dense(7, Softmax)     │ → Output: 7 probabilitas emosi
└────────────────────────┘
```

---

## ❓ Error Umum & Solusi

### 1. `ModuleNotFoundError: No module named 'tensorflow'`
**Solusi:** Install ulang requirements
```bash
pip install -r requirements.txt
```

### 2. `Cannot open camera` atau webcam tidak terbuka
**Solusi:**
- Pastikan webcam tersambung
- Tutup aplikasi lain yang menggunakan webcam (Zoom, Teams, dll)
- Coba ganti `cv2.VideoCapture(0)` dengan `cv2.VideoCapture(1)` di `realtime_detection.py`

### 3. `ERROR: Dataset belum disiapkan!`
**Solusi:** Jalankan konversi dataset dulu:
```bash
python src/prepare_dataset.py
```

### 4. `ERROR: Model tidak ditemukan`
**Solusi:** Jalankan training dulu:
```bash
python src/train.py
```

### 5. Training sangat lambat
**Solusi:**
- Kurangi `EPOCHS` di `config.py` (misalnya dari 50 ke 25)
- Kurangi `BATCH_SIZE` jika kehabisan memori
- Gunakan GPU jika tersedia (install `tensorflow-gpu`)

### 6. `execution policy` error di PowerShell
**Solusi:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 7. Akurasi rendah (<50%)
**Kemungkinan penyebab:**
- Dataset belum terkonversi dengan benar → jalankan ulang `prepare_dataset.py`
- Training belum cukup lama → tambah `EPOCHS` di `config.py`
- Coba tambah data augmentasi di `utils.py`

---

## 📊 Hasil yang Diharapkan

Setelah training dan evaluasi, folder `results/` akan berisi:

| File | Deskripsi |
|------|-----------|
| `training_history.png` | Grafik accuracy & loss selama training |
| `confusion_matrix.png` | Matrix kesalahan prediksi per kelas |
| `classification_report.txt` | Precision, recall, F1-score per kelas |
| `training_history.json` | Data history training (format JSON) |
| `screenshot_*.png` | Screenshot dari deteksi real-time |

---

## 📝 Catatan Teknis

- Model menggunakan **CNN custom** (bukan transfer learning)
- Input gambar berformat **grayscale 48x48 piksel**
- Normalisasi piksel ke range **0-1**
- Augmentasi data: rotasi, flip, zoom, shift, shear
- Optimizer: **Adam** dengan learning rate awal 0.001
- Loss function: **Categorical Crossentropy**
- Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

---

## 🔧 Kustomisasi

Semua parameter bisa diubah di file `src/config.py`:

```python
BATCH_SIZE = 64       # Ubah batch size
EPOCHS = 50           # Ubah jumlah epoch
LEARNING_RATE = 0.001 # Ubah learning rate
IMG_SIZE = 48         # Ukuran gambar (jangan diubah untuk FER-2013)
```

---

*Dibuat sebagai project pembelajaran Machine Learning — Facial Emotion Recognition dengan CNN & FER-2013*
