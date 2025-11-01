import os
import cv2
import albumentations as A
import random
import shutil

# Path
input_images_folder = 'Data Baru/images'  # folder gambar
input_labels_folder = 'Data Baru/labels'  # folder labels .txt
output_images_folder = 'augmentasi/images'
output_labels_folder = 'augmentasi/labels'

# Buat folder output kalau belum ada
os.makedirs(output_images_folder, exist_ok=True)
os.makedirs(output_labels_folder, exist_ok=True)

# Augmentasi yang akan digunakan
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Fungsi load bounding box
def load_labels(label_path):
    boxes = []
    class_labels = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                cls, x_center, y_center, width, height = parts
                boxes.append([float(x_center), float(y_center), float(width), float(height)])
                class_labels.append(int(cls))
    return boxes, class_labels

# Fungsi save bounding box
def save_labels(label_path, boxes, class_labels):
    with open(label_path, 'w') as f:
        for box, cls in zip(boxes, class_labels):
            f.write(f"{cls} {box[0]} {box[1]} {box[2]} {box[3]}\n")

# Proses semua gambar
for filename in os.listdir(input_images_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_images_folder, filename)
        label_path = os.path.join(input_labels_folder, filename.rsplit('.', 1)[0] + '.txt')

        if not os.path.exists(label_path):
            print(f"Label tidak ditemukan untuk {filename}, skip...")
            continue

        image = cv2.imread(image_path)
        height, width, _ = image.shape

        boxes, class_labels = load_labels(label_path)

        # Lakukan augmentasi beberapa kali (contoh: 3 augmentasi per gambar)
        for i in range(3):
            augmented = transform(image=image, bboxes=boxes, class_labels=class_labels)
            aug_image = augmented['image']
            aug_boxes = augmented['bboxes']
            aug_class_labels = augmented['class_labels']

            # Save gambar baru
            new_filename = filename.rsplit('.', 1)[0] + f'_aug{i}.jpg'
            new_labelname = filename.rsplit('.', 1)[0] + f'_aug{i}.txt'

            cv2.imwrite(os.path.join(output_images_folder, new_filename), aug_image)
            save_labels(os.path.join(output_labels_folder, new_labelname), aug_boxes, aug_class_labels)

        # Copy gambar original + label ke output (optional)
        shutil.copy(image_path, os.path.join(output_images_folder, filename))
        shutil.copy(label_path, os.path.join(output_labels_folder, filename.rsplit('.', 1)[0] + '.txt'))

print("Augmentasi selesai! 🚀")
