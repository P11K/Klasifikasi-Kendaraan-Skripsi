from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Load model dan lakukan evaluasi
model = YOLO('yolov8m.pt')
results = model.val()

# Kelas target yang ingin dimasukkan
target_classes = [5, 3, 7, 2]  # motor, mobil, truk

# Ambil nama class untuk label
class_names = model.names

# Simpan label ground truth dan prediksi
true_labels = []
pred_labels = []

for pred in results:
    for det in pred.boxes:
        cls = int(det.cls.cpu().numpy())
        if cls in target_classes:
            true_labels.append(cls)
            pred_labels.append(cls)  # Kalau kamu punya mapping prediksi → ground truth

# Konversi label ke index relatif (bukan ke ID asli)
id_to_idx = {id_: i for i, id_ in enumerate(target_classes)}
true_idx = [id_to_idx[l] for l in true_labels]
pred_idx = [id_to_idx[l] for l in pred_labels]

# Buat confusion matrix
cm = confusion_matrix(true_idx, pred_idx)
display_labels = [class_names[i] for i in target_classes]

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix (Motor, Mobil, Truk)")
plt.show()
