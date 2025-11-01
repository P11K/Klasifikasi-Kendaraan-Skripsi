import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm

# --- KONFIGURASI ---

MODEL_PATH = 'yolov8x.pt'  # Path ke model YOLOv8
IMAGE_FOLDER = "Anotasi Pagi/images"  # Folder input gambar
OUTPUT_FOLDER = 'visualisasi_truck'  # Folder untuk hasil
TARGET_CLASS = 'truck'
CONFIDENCE_THRESHOLD = 0.25

# --- CEK PATH ---

if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: File model tidak ditemukan di '{MODEL_PATH}'")
    exit()

if not os.path.exists(IMAGE_FOLDER):
    print(f"❌ Error: Folder gambar tidak ditemukan di '{IMAGE_FOLDER}'")
    exit()

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model
print("🚀 Memuat model...")
model = YOLO(MODEL_PATH)

# Ambil file gambar dari folder
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if not image_files:
    print(f"⚠️ Tidak ada gambar ditemukan di '{IMAGE_FOLDER}'")
    exit()

print(f"🎯 Memproses hanya gambar yang mengandung kelas '{TARGET_CLASS}'...")
for img_name in tqdm(image_files, desc="🔍 Memeriksa gambar"):
    img_path = os.path.join(IMAGE_FOLDER, img_name)
    image = cv2.imread(img_path)
    if image is None:
        print(f"\n⚠️ Gagal membaca gambar {img_name}, dilewati.")
        continue

    results = model(image, conf=CONFIDENCE_THRESHOLD, verbose=False)

    truck_detected = False
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id].lower()

            if class_name == TARGET_CLASS:
                truck_detected = True
                break  # cukup satu truck, tidak perlu lanjut

    if truck_detected:
        output_path = os.path.join(OUTPUT_FOLDER, img_name)
        cv2.imwrite(output_path, image)

print(f"\n✅ Selesai. Gambar yang mengandung '{TARGET_CLASS}' disalin ke: {OUTPUT_FOLDER}")
