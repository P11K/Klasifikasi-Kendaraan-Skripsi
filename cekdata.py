import os
from collections import Counter

label_folder = "dataset_split/train/labels"  # Ganti sesuai path folder label kamu
label_counts = Counter()

for file in os.listdir(label_folder):
    if file.endswith(".txt"):
        with open(os.path.join(label_folder, file), 'r') as f:
            for line in f:
                class_id = line.strip().split()[0]
                label_counts[class_id] += 1

# Tampilkan hasilnya
for class_id, count in sorted(label_counts.items(), key=lambda x: int(x[0])):
    print(f"Kelas {class_id}: {count} instance")
