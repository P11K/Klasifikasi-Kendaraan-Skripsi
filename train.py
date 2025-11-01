from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import os

def main():
    # Load model
    model = YOLO('runs/detect/Finall/weights/last.pt')  # atau 'yolov8l.pt'

    # Train model
    model.train(
        data='datamix.yaml',
        epochs=100,                   # Jumlah epoch
        imgsz=640,                    # Ukuran gambar
        batch=8,                     # Batch size (bisa dikurangi kalau RAM/GPU kecil)
        name='Finall',        # Nama folder hasil training
        pretrained=True,              # Gunakan bobot pretrained COCO
        device=0,                     # Gunakan GPU (0) atau CPU (-1)
        workers=8,                    # Jumlah worker untuk data loading
        save=True,                    # Simpan model di setiap epoch terbaik
        val=True,
        lr0=0.001  # Default: 0.01 → Coba kecilkan jadi 0.001 atau 0.0005
        # resume=True
    )

if __name__ == '__main__':
    main()
