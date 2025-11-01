from ultralytics import YOLO

model = YOLO('runs/train/maneh/weights/best.pt')  # Ganti dengan path model Anda
def main():
    model.train(
        data="datamix.yaml",
        epochs=40,
        imgsz=640,
        batch=14,
        device="0",
        workers=2,
        single_cls=False,
        # Parameter yang Diperbaiki:
        rect=False,           # Ganti "persist" dengan "rect" (jika perlu)
        lr0=8e-5,
        lrf=0.01,
        optimizer="AdamW",
        weight_decay=0.05,
        momentum=0.9,
        freeze=[0,1,2,3,4,5],
        augment=True,
        mosaic=0.5,
        mixup=0.0,
        hsv_h=0.2,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=5.0,
        flipud=0.1,
        shear=1.5,
        name="fine",
        project="runs/detect",
        exist_ok=True,
        patience=10,
        # Perbaikan untuk Loss Weights (gunakan parameter terpisah):
        box=6.0,    # Loss weight untuk bounding box
        cls=1.2,    # Loss weight untuk klasifikasi
        dfl=1.5     # Loss weight untuk Distribution Focal Loss
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
                                                # fine tuning bengi

# Batasi jumlah thread jika perlu
# torch.set_num_threads(4)
# os.environ["OMP_NUM_THREADS"] = "4"

# # Load model hasil training sebelumnya
# model = YOLO("runs/train/yolo-brlg/weights/best.pt")

# def main():
#     model.train(
#         data="finetuning.yaml",     # Dataset campuran siang & malam
#         epochs=25,
#         imgsz=704,
#         batch=14,
#         device="0",
#         name="mix",
#         project="runs/train",
#         exist_ok=True,

#         # Fine-tuning setup
#         lr0=0.0001,
#         lrf=0.01,

#         # Freeze backbone layer awal untuk stabilitas di awal training
#         freeze=[0, 1],   # Nanti bisa dilonggarkan manual setelah 20 epoch kalau pakai dua tahap training

#         # Augmentasi ringan tapi adaptif
#         translate=0.1,
#         scale=0.3,
#         hsv_h=0.015,     # Hue shift kecil
#         hsv_s=0.7,       # Saturasi naik
#         hsv_v=0.4,       # Brightness/kontras naik sedikit

#         patience=10,
#         verbose=True,
#         augment=True
#     )

# if __name__ == "__main__":
#     import multiprocessing
#     multiprocessing.freeze_support()
#     main()