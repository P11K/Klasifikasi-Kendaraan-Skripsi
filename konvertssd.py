import os
from glob import glob
import xml.etree.ElementTree as ET
from PIL import Image

# Mapping dari index ke nama kelas
CLASS_NAMES = ['Bus', 'Motor', 'Truck', 'mobil']  # Sesuaikan dengan dataset kamu

def create_voc_xml(folder, filename, image_size, objects):
    annotation = ET.Element("annotation")
    
    ET.SubElement(annotation, "folder").text = folder
    ET.SubElement(annotation, "filename").text = filename

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(image_size[0])
    ET.SubElement(size, "height").text = str(image_size[1])
    ET.SubElement(size, "depth").text = "3"

    for obj in objects:
        name, xmin, ymin, xmax, ymax = obj
        object_item = ET.SubElement(annotation, "object")
        ET.SubElement(object_item, "name").text = name
        ET.SubElement(object_item, "pose").text = "Unspecified"
        ET.SubElement(object_item, "truncated").text = "0"
        ET.SubElement(object_item, "difficult").text = "0"
        bbox = ET.SubElement(object_item, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(xmin)
        ET.SubElement(bbox, "ymin").text = str(ymin)
        ET.SubElement(bbox, "xmax").text = str(xmax)
        ET.SubElement(bbox, "ymax").text = str(ymax)

    return ET.ElementTree(annotation)

def convert_yolo_to_voc(image_path, label_path, save_path, folder_name):
    img = Image.open(image_path)
    width, height = img.size
    filename = os.path.basename(image_path)

    objects = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                xmin = int((cx - w / 2) * width)
                ymin = int((cy - h / 2) * height)
                xmax = int((cx + w / 2) * width)
                ymax = int((cy + h / 2) * height)
                label = CLASS_NAMES[class_id]
                objects.append((label, xmin, ymin, xmax, ymax))

    xml_tree = create_voc_xml(folder_name, filename, (width, height), objects)
    os.makedirs(save_path, exist_ok=True)
    xml_name = os.path.splitext(filename)[0] + ".xml"
    xml_tree.write(os.path.join(save_path, xml_name))

def process_split(split):
    img_dir = f"dataset_split/{split}/images"
    label_dir = f"dataset_split/{split}/labels"
    save_dir = f"voc/{split}/Annotations"
    os.makedirs(save_dir, exist_ok=True)

    image_paths = glob(os.path.join(img_dir, "*.jpg"))
    if not image_paths:
        image_paths = glob(os.path.join(img_dir, "*.png"))  # Jaga-jaga

    for img_path in image_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + ".txt")
        convert_yolo_to_voc(img_path, label_path, save_dir, split)

# Proses semua split
for split in ['train', 'val', 'test']:
    process_split(split)

print("✅ Konversi selesai ke folder 'voc/train/Annotations', dst.")
