import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Inisialisasi model YOLOv8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('runs/train/fine_tuned_model/weights/best.pt').to(device)

# Konfigurasi class kendaraan
vehicle_classes = {
    0: ("Bus", (0, 255, 0)),
    1: ("Motor", (255, 0, 0)),
    2: ("Truck", (0, 255, 255)),
    3: ("Mobil", (255, 0, 255))
}
counter = {name: 0 for _, (name, _) in vehicle_classes.items()}

# Inisialisasi video
video_path = r"C:\Users\Piko\Videos\Captures\datamotor.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"Tidak dapat membuka video: {video_path}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Variabel untuk menggambar ROI
roi_points = []
drawing_roi = False
current_temp_point = None

def draw_roi(event, x, y, flags, param):
    global roi_points, drawing_roi, current_temp_point
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_roi = True
        roi_points.append((x, y))
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing_roi:
            current_temp_point = (x, y)
            
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(roi_points) > 0:
            drawing_roi = False
            current_temp_point = None

# Pemilihan ROI pada frame pertama
ret, first_frame = cap.read()
if not ret:
    raise ValueError("Gagal membaca frame pertama")

cv2.namedWindow("Gambar ROI (Kiri: Tambah Titik, Kanan: Selesai)")
cv2.setMouseCallback("Gambar ROI (Kiri: Tambah Titik, Kanan: Selesai)", draw_roi)

while True:
    temp_frame = first_frame.copy()
    
    # Gambar garis ROI
    if len(roi_points) > 0:
        for i in range(len(roi_points)-1):
            cv2.line(temp_frame, roi_points[i], roi_points[i+1], (0,0,255), 2)
        if current_temp_point:
            cv2.line(temp_frame, roi_points[-1], current_temp_point, (0,0,255), 2)
            
    cv2.imshow("Gambar ROI (Kiri: Tambah Titik, Kanan: Selesai)", temp_frame)
    key = cv2.waitKey(1)
    
    if key == ord(' '):  # Spasi untuk konfirmasi
        if len(roi_points) >= 2:
            break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Fungsi deteksi persilangan garis
def line_intersection(line1, line2):
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]
    
    den = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if den == 0:
        return None
    
    t = ((x1 - x3)*(y3 - y4) - (y1 - y3)*(x3 - x4)) / den
    u = -((x1 - x2)*(y1 - y3) - (y1 - y2)*(x1 - x3)) / den
    
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (int(x1 + t*(x2 - x1)), int(y1 + t*(y2 - y1)))
    return None

# Inisialisasi video writer
output_path = "output.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print(f"ROI Line Points: {roi_points}")

track_history = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        conf=0.3,
        iou=0.5,
        classes=list(vehicle_classes.keys()),
        persist=True,
        verbose=False
    )

    # Gambar garis ROI
    if len(roi_points) >= 2:
        for i in range(len(roi_points)-1):
            cv2.line(frame, roi_points[i], roi_points[i+1], (0,255,255), 2)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, class_id, track_id in zip(boxes, class_ids, track_ids):
            if class_id not in vehicle_classes:
                continue

            class_name, color = vehicle_classes[class_id]
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2  # Tetap di tengah horizontal
            cy = y2  # Ambil nilai y2 (bawah bounding box)

            # Gambar bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, class_name, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Gambar titik di bagian bawah
            cv2.circle(frame, (cx, cy), 5, color, -1)  # Perhatikan cy = y2

                # Tracking posisi
            if track_id not in track_history:
                track_history[track_id] = {
                    'positions': [],
                    'counted': False
                }
                
            current_pos = (cx, cy)  # Menggunakan cy = y2
            
            if len(track_history[track_id]['positions']) > 0:
                prev_pos = track_history[track_id]['positions'][-1]
                
                # Cek persilangan dengan semua segmen ROI
                for i in range(len(roi_points)-1):
                    roi_line = (roi_points[i], roi_points[i+1])
                    vehicle_line = (prev_pos, current_pos)
                    
                    intersect = line_intersection(roi_line, vehicle_line)
                    if intersect and not track_history[track_id]['counted']:
                        counter[class_name] += 1
                        track_history[track_id]['counted'] = True
                        cv2.circle(frame, intersect, 7, (0,0,255), -1)
                        print(f"{class_name} (ID {track_id}) menyebrangi ROI di {intersect}")
                        break
                else:
                    track_history[track_id]['counted'] = False
                    
            track_history[track_id]['positions'].append(current_pos)

    # Tampilkan counter
    y_pos = 30
    for class_name, count in counter.items():
        cv2.putText(frame, f"{class_name}: {count}", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30

    cv2.imshow("Vehicle Detection & Counting", frame)
    video_writer.write(frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print("\n--- Hasil Akhir ---")
for class_name, count in counter.items():
    print(f"{class_name}: {count}")