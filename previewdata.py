from ultralytics import YOLO
import cv2

# Load model YOLOv8x pretrained COCO
model = YOLO("yolov8x.pt")
# model = YOLO("runs/train/bengi/weights/best.pt")  # Ganti path sesuai modelmu

# Hanya deteksi motor
wanted_classes = ['truck','bus','car','motorcycle']
wanted_class_ids = [cls_id for cls_id, name in model.names.items() if name in wanted_classes]

# Threshold minimum confidence
conf_threshold = 0.6

# Load video
video_path = r"C:\Users\Piko\Videos\Captures\dataset.mp4"  # Ganti dengan path ke video kamu
cap = cv2.VideoCapture(video_path)

# Ukuran layar
screen_width = 1280
screen_height = 720

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Deteksi
    results = model(frame, verbose=False)[0]

    # Copy frame untuk anotasi
    annotated_frame = frame.copy()
    frame_confidences = []

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        conf = box.conf[0].item()

        if cls_id in wanted_class_ids and conf >= conf_threshold:
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Simpan confidence untuk logging
            frame_confidences.append(conf)

            # Gambar bounding box dan label
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Print confidence per frame (jika ada)
    if frame_confidences:
        avg_conf = sum(frame_confidences) / len(frame_confidences)
        print(f'Frame {frame_count} - Detected {len(frame_confidences)} motor(s), Avg Confidence: {avg_conf:.2f}')
    else:
        print(f'Frame {frame_count} - No motorcycle detected.')

    # Resize & tampilkan
    resized_frame = cv2.resize(annotated_frame, (screen_width, screen_height))
    cv2.imshow("Deteksi Kendaraan", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
