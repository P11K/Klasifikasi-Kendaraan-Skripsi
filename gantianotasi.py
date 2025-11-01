import os

# Folder tempat label .txt hasil anotasi baru kamu
label_folder = r"C:\Users\Piko\Documents\Tugas Kuliah\Skripsi\Yolo\Test Pagi\valid\labels"

# Mapping: dari ID lama ke ID baru (sesuai model)
id_mapping = {
    0: 0,  # Mobil → mobil
    1: 3,  # Motor → Motor
    2: 1,
    3: 2
}

for filename in os.listdir(label_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(label_folder, filename)
        with open(file_path, "r") as file:
            lines = file.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # skip invalid line
            old_id = int(parts[0])
            new_id = id_mapping.get(old_id, old_id)  # ganti ID jika ada di mapping
            parts[0] = str(new_id)
            new_lines.append(" ".join(parts))

        with open(file_path, "w") as file:
            file.write("\n".join(new_lines))
