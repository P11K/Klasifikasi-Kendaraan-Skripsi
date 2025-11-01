from ultralytics import YOLO
import os
import cv2
from tqdm import tqdm

# --- Pengaturan Folder ---
main_folder = "Anotasi Malam"
input_folder = os.path.join(main_folder, "images")
output_label_folder = os.path.join(main_folder, "labels")

os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)
# -------------------------

# Load dua model
model_general = YOLO("yolov8x.pt")  # Untuk bus, motorcycle, car
model_truck = YOLO("runs/detect/fine/weights/best.pt")  # Hanya untuk truck

# Daftar kelas target dan urutan ID
wanted_classes = ['bus', 'motorcycle', 'truck', 'car']
confidence_threshold = 0.5

# Daftar gambar
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
if not image_files:
    print(f"⚠️  Tidak ada gambar ditemukan di '{input_folder}'.")
    exit()

print(f"📸 Mulai anotasi {len(image_files)} gambar...")

for img_name in tqdm(image_files, desc="Memproses Gambar"):
    img_path = os.path.join(input_folder, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️  Gagal membaca gambar {img_name}, dilewati.")
        continue
    h, w, _ = img.shape

    label_path = os.path.join(output_label_folder, os.path.splitext(img_name)[0] + ".txt")

    with open(label_path, "w") as f:

        # --- Model Umum: bus, motorcycle, car ---
        results_general = model_general(img_path, verbose=False)
        for result in results_general:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                class_name = model_general.names[cls_id].lower()
                conf = box.conf[0].item()
                if conf >= confidence_threshold and class_name in ['bus', 'motorcycle', 'car']:
                    new_cls_id = wanted_classes.index(class_name)
                    x_center, y_center, bw, bh = box.xywhn[0].tolist()
                    f.write(f"{new_cls_id} {x_center} {y_center} {bw} {bh}\n")

        # --- Model Truck Khusus ---
        results_truck = model_truck(img_path, verbose=False)
        for result in results_truck:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                class_name = model_truck.names[cls_id].lower()
                conf = box.conf[0].item()
                if conf >= confidence_threshold and class_name == 'truck':
                    new_cls_id = wanted_classes.index('truck')
                    x_center, y_center, bw, bh = box.xywhn[0].tolist()
                    f.write(f"{new_cls_id} {x_center} {y_center} {bw} {bh}\n")

print(f"\n✅ Selesai! File anotasi disimpan di '{output_label_folder}'")
