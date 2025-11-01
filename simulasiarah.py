import cv2
import math
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("runs/detect/fine/weights/best.pt")

# Tracker posisi sebelumnya
previous_positions = {}

# Fungsi arah + sudut
def get_direction_and_angle(prev, curr):
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]
    angle = math.degrees(math.atan2(-dy, dx))
    jarak = math.hypot(dx, dy)

    arah = "diam"
    if jarak > 5:
        if 30 <= angle < 90:
            arah = "bawah"
        elif -45 <= angle <= 45:
            arah = "kanan"
        elif -110 < angle <= -90:
            arah = "atas"
        elif angle > 70 or angle < -90:
            arah = "kiri"
    return arah, angle

# Load video
cap = cv2.VideoCapture(r"C:\Users\Piko\Videos\Captures\data test.mp4")

# Set fullscreen window
cv2.namedWindow("Arah + Sudut + Fullscreen", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Arah + Sudut + Fullscreen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Target tampilan layar penuh
screen_width = 1920  # atau bisa pakai pyautogui.size()
screen_height = 1080

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    original_h, original_w = frame.shape[:2]
    scale_w = screen_width / original_w
    scale_h = screen_height / original_h
    scale = min(scale_w, scale_h)

    # Resize ke layar
    new_w, new_h = int(original_w * scale), int(original_h * scale)
    frame = cv2.resize(frame, (new_w, new_h))

    mid_x, mid_y = new_w // 2, new_h // 2
    cv2.line(frame, (mid_x, 0), (mid_x, new_h), (200, 200, 200), 1)
    cv2.line(frame, (0, mid_y), (new_w, mid_y), (200, 200, 200), 1)
    cv2.putText(frame, "X+", (new_w - 40, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, "Y+", (mid_x + 10, new_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    results = model.track(frame, persist=True, conf=0.4, iou=0.5, verbose=False)

    if results and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, obj_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, y2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            if obj_id in previous_positions:
                prev_cx, prev_cy = previous_positions[obj_id]
                arah, angle = get_direction_and_angle((prev_cx, prev_cy), (cx, cy))
                cv2.arrowedLine(frame, (prev_cx, prev_cy), (cx, cy), (255, 0, 0), 2, tipLength=0.3)
                label_text = f"{arah} ({angle:.1f}°)"
                cv2.putText(frame, label_text, (cx + 5, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            previous_positions[obj_id] = (cx, cy)

    cv2.imshow("Arah + Sudut + Fullscreen", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
