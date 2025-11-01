from ultralytics import YOLO
import cv2
import time

# Load model hasil training
model = YOLO("runs/train/mix/weights/best.pt")  # Ganti path sesuai modelmu
# model = YOLO("yolov8x.pt")  # Ganti path sesuai modelmu

# Load video input
video_path = r"C:\Users\Piko\Videos\Captures\24.mp4"  # Ganti dengan path ke video kamu
cap = cv2.VideoCapture(video_path)

# Ambil ukuran layar (width, height)
screen_width = 1920
screen_height = 1080

prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # Deteksi kendaraan di frame
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    # Hitung FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # Tampilkan FPS di frame
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Resize frame agar sesuai dengan ukuran layar
    resized_frame = cv2.resize(annotated_frame, (screen_width, screen_height))

    # Tampilkan frame
    cv2.imshow("Deteksi Kendaraan", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()
