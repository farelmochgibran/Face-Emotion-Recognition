"""
=================================================================
prepare_dataset.py â€” Konversi Dataset Arrow ke Folder Gambar
=================================================================
Script ini mengkonversi dataset FER-2013 dari format HuggingFace
Arrow menjadi folder gambar per kelas emosi.

Format output:
  dataset/fer2013/
  â”œâ”€â”€ train/
  â”‚   â”œâ”€â”€ angry/
  â”‚   â”‚   â”œâ”€â”€ 00001.png
  â”‚   â”‚   â””â”€â”€ ...
  â”‚   â”œâ”€â”€ disgust/
  â”‚   â””â”€â”€ ...
  â””â”€â”€ test/
      â”œâ”€â”€ angry/
      â””â”€â”€ ...

Jalankan script ini SATU KALI sebelum training:
  python src/prepare_dataset.py
=================================================================
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image

# Tambahkan parent directory ke path agar bisa import config
sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    ARROW_DATASET_DIR, DATASET_DIR, TRAIN_DIR, TEST_DIR,
    EMOTION_LABELS, IMG_SIZE
)


def convert_arrow_to_images():
    """
    Membaca dataset Arrow dari HuggingFace dan menyimpan
    setiap gambar sebagai file PNG di folder per kelas emosi.
    """
    
    # Import library datasets dari HuggingFace
    try:
        from datasets import load_from_disk
    except ImportError:
        print("ERROR: Library 'datasets' belum terinstall!")
        print("Jalankan: pip install datasets")
        sys.exit(1)
    
    # Cek apakah folder dataset Arrow ada
    if not ARROW_DATASET_DIR.exists():
        print(f"ERROR: Folder dataset tidak ditemukan di:")
        print(f"  {ARROW_DATASET_DIR}")
        print()
        print("Pastikan folder 'fer2013_enhanced' ada di lokasi yang benar.")
        sys.exit(1)
    
    print("=" * 60)
    print("  KONVERSI DATASET ARROW KE FOLDER GAMBAR")
    print("=" * 60)
    
    # Load dataset dari disk
    print("\n[1/3] Memuat dataset dari disk...")
    dataset = load_from_disk(str(ARROW_DATASET_DIR))
    print(f"  Dataset berhasil dimuat!")
    print(f"  Splits yang tersedia: {list(dataset.keys())}")
    
    # Mapping split name ke folder output
    # 'train' -> train, 'test' -> test, 'validation' -> test (digabung)
    split_mapping = {
        "train": TRAIN_DIR,
        "test": TEST_DIR,
    }
    
    # Buat folder untuk setiap kelas di setiap split
    print("\n[2/3] Membuat struktur folder...")
    emotion_names = [e.lower() for e in EMOTION_LABELS]
    
    for split_name, split_dir in split_mapping.items():
        for emotion in emotion_names:
            folder = split_dir / emotion
            folder.mkdir(parents=True, exist_ok=True)
            print(f"  + {folder}")
    
    # Konversi dan simpan gambar
    print("\n[3/3] Menyimpan gambar ke folder...")
    
    for split_name, split_dir in split_mapping.items():
        if split_name not in dataset:
            print(f"  ! Split '{split_name}' tidak ditemukan, skip.")
            continue
        
        split_data = dataset[split_name]
        total = len(split_data)
        counters = {e: 0 for e in emotion_names}
        
        print(f"\n  Memproses split '{split_name}' ({total} gambar)...")
        
        for i, sample in enumerate(split_data):
            # Ambil data gambar dan label
            image_array = np.array(sample["image"], dtype=np.uint8)
            emotion_idx = sample["emotion"]
            emotion_name = emotion_names[emotion_idx]
            
            # Pastikan ukuran gambar 48x48
            if image_array.shape != (IMG_SIZE, IMG_SIZE):
                # Resize jika perlu
                img = Image.fromarray(image_array, mode='L')
                img = img.resize((IMG_SIZE, IMG_SIZE))
            else:
                img = Image.fromarray(image_array, mode='L')
            
            # Simpan gambar
            counters[emotion_name] += 1
            filename = f"{counters[emotion_name]:05d}.png"
            save_path = split_dir / emotion_name / filename
            img.save(save_path)
            
            # Progress bar sederhana
            if (i + 1) % 1000 == 0 or (i + 1) == total:
                print(f"    Progres: {i+1}/{total} "
                      f"({(i+1)/total*100:.1f}%)")
        
        # Tampilkan ringkasan per kelas
        print(f"\n  Ringkasan split '{split_name}':")
        for emotion, count in counters.items():
            print(f"    {emotion:>10s}: {count:>5d} gambar")
        print(f"    {'TOTAL':>10s}: {sum(counters.values()):>5d} gambar")
    
    # Jika ada split validation, gabungkan ke test
    if "validation" in dataset:
        print(f"\n  Memproses split 'validation' (digabung ke test)...")
        val_data = dataset["validation"]
        total = len(val_data)
        
        # Hitung offset dari gambar test yang sudah ada
        existing_counts = {}
        for emotion in emotion_names:
            folder = TEST_DIR / emotion
            existing = len(list(folder.glob("*.png")))
            existing_counts[emotion] = existing
        
        counters = {e: existing_counts[e] for e in emotion_names}
        
        for i, sample in enumerate(val_data):
            image_array = np.array(sample["image"], dtype=np.uint8)
            emotion_idx = sample["emotion"]
            emotion_name = emotion_names[emotion_idx]
            
            if image_array.shape != (IMG_SIZE, IMG_SIZE):
                img = Image.fromarray(image_array, mode='L')
                img = img.resize((IMG_SIZE, IMG_SIZE))
            else:
                img = Image.fromarray(image_array, mode='L')
            
            counters[emotion_name] += 1
            filename = f"{counters[emotion_name]:05d}.png"
            save_path = TEST_DIR / emotion_name / filename
            img.save(save_path)
            
            if (i + 1) % 1000 == 0 or (i + 1) == total:
                print(f"    Progres: {i+1}/{total} "
                      f"({(i+1)/total*100:.1f}%)")
        
        added = {e: counters[e] - existing_counts[e] for e in emotion_names}
        print(f"\n  Validation ditambahkan ke test:")
        for emotion, count in added.items():
            print(f"    {emotion:>10s}: +{count:>4d} gambar")
    
    print("\n" + "=" * 60)
    print("  KONVERSI SELESAI!")
    print("=" * 60)
    print(f"\n  Dataset tersimpan di: {DATASET_DIR}")
    print(f"  Selanjutnya jalankan: python src/train.py")
    print()


if __name__ == "__main__":
    convert_arrow_to_images()
