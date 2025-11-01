import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm

# --- KONFIGURASI ---

MODEL_PATH = 'runs/detect/fine/weights/best.pt'  # Ganti path model
IMAGE_FOLDER = 'visualisasi_truck'  # Ganti folder gambar
ANNOTATION_OUTPUT = 'visualisasi_truck/labels'  # Folder untuk file anotasi .txt
TARGET_CLASS = 'truck'
TARGET_CLASS_ID = 2  # ID class untuk 'truck' yang akan ditulis di file .txt
CONFIDENCE_THRESHOLD = 0.25

# --- CEK PATH ---

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model tidak ditemukan di: {MODEL_PATH}")
    exit()

if not os.path.exists(IMAGE_FOLDER):
    print(f"❌ Folder gambar tidak ditemukan di: {IMAGE_FOLDER}")
    exit()

os.makedirs(ANNOTATION_OUTPUT, exist_ok=True)

# Load model
print("🚀 Memuat model...")
model = YOLO(MODEL_PATH)

# Ambil file gambar
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if not image_files:
    print(f"⚠️ Tidak ada gambar ditemukan di: {IMAGE_FOLDER}")
    exit()

print(f"📝 Membuat anotasi YOLO untuk kelas '{TARGET_CLASS}' dengan ID {TARGET_CLASS_ID}")
for img_name in tqdm(image_files, desc="🔍 Memproses anotasi"):
    img_path = os.path.join(IMAGE_FOLDER, img_name)
    image = cv2.imread(img_path)

    if image is None:
        print(f"\n⚠️ Gagal membaca gambar {img_name}, dilewati.")
        continue

    h, w = image.shape[:2]
    results = model(image, conf=CONFIDENCE_THRESHOLD, verbose=False)

    annotations = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id].lower()

            if class_name == TARGET_CLASS:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                # Hitung koordinat dalam format relatif YOLO
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                box_width = (x2 - x1) / w
                box_height = (y2 - y1) / h

                # Format: class_id x_center y_center width height
                annotations.append(f"{TARGET_CLASS_ID} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    # Simpan file .txt jika ada deteksi
    if annotations:
        file_name = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(ANNOTATION_OUTPUT, file_name)
        with open(txt_path, 'w') as f:
            f.write("\n".join(annotations))

print(f"\n✅ Anotasi selesai. Hasil disimpan di folder: {ANNOTATION_OUTPUT}")
