import cv2
import torch
from ultralytics import YOLO
from tkinter import Tk, filedialog
from collections import defaultdict
import numpy as np

# Pastikan CUDA tersedia
if not torch.cuda.is_available():
    print("CUDA tidak tersedia.")
    exit()

# Load model YOLOv8 ke GPU
model = YOLO("runs/detect/fine/weights/best.pt")  # Ganti path model Anda
model.to("cuda")

# Daftar nama kelas
class_names = ["Bus", "Motor", "Truck", "Mobil"]

# Pilih video melalui file explorer
Tk().withdraw()
video_path = filedialog.askopenfilename(title="Pilih Video", filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
if not video_path:
    print("Tidak ada video dipilih.")
    exit()

# Buka video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Gagal membuka video.")
    exit()

# Hitung jumlah deteksi dan confidence per kelas
class_counts = defaultdict(int)
confidence_totals = defaultdict(float)
confidence_counts = defaultdict(int)
total_detections = 0

# Target tinggi video
target_height = 480
min_confidence_threshold = 0.6  # Minimum confidence 70%

last_frame = None  # Simpan frame terakhir

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    scale = target_height / height
    resized_frame = cv2.resize(frame, (int(width * scale), target_height))

    results = model(resized_frame, verbose=False, device='cuda')[0]

    if results.boxes is not None:
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()
        for cls_id, conf in zip(classes, confidences):
            if conf >= min_confidence_threshold and 0 <= cls_id < len(class_names):
                class_name = class_names[cls_id]
                class_counts[class_name] += 1
                confidence_totals[class_name] += conf
                confidence_counts[class_name] += 1
                total_detections += 1

    annotated_frame = results.plot()

    # Panel teks kanan
    h, w, _ = annotated_frame.shape
    panel_width = 250
    text_panel = np.zeros((h, panel_width, 3), dtype=np.uint8)

    cv2.putText(text_panel, "Confidence per Kelas:", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    for i, class_name in enumerate(class_names):
        count = confidence_counts[class_name]
        avg_conf = (confidence_totals[class_name] / count * 100) if count > 0 else 0
        text = f"{class_name}: {avg_conf:.1f}%"
        cv2.putText(text_panel, text, (10, 70 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    combined_frame = np.hstack((annotated_frame, text_panel))
    last_frame = combined_frame.copy()

    cv2.imshow("Deteksi Kendaraan (YOLOv8 + CUDA)", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Tampilkan hasil confidence rata-rata di terminal
print("\n--- Rata-rata Confidence per Kelas (min 70%) ---")
if total_detections > 0:
    for class_name in class_names:
        count = confidence_counts[class_name]
        avg_conf = (confidence_totals[class_name] / count * 100) if count > 0 else 0
        print(f"{class_name}: {avg_conf:.2f}%")
else:
    print("Tidak ada deteksi dengan confidence >= 70% selama video berlangsung.")

# Tampilkan frame terakhir dengan notifikasi, tunggu tombol 'w'
if last_frame is not None:
    notif_panel = np.zeros((target_height, 300, 3), dtype=np.uint8)
    cv2.putText(notif_panel, "Video selesai", (30, 100), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)
    cv2.putText(notif_panel, "Tekan 'w' untuk keluar", (30, 150), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 1)
    final_display = np.hstack((last_frame, notif_panel))

    while True:
        cv2.imshow("Selesai", final_display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('w'):
            break

cv2.destroyAllWindows()
