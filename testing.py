from ultralytics import YOLO

# Load model
model = YOLO('runs/detect/fine/weights/best.pt')  # Ganti path model kamu

def eval():
    # Jalankan evaluasi
    results = model.val(
        data="datamix.yaml",
        imgsz=640,
        conf=0.25,
        iou=0.5,
        split="val"
    )
    # Ambil nama kelas dari model
    class_names = model.names

    # Ambil precision (p), recall (r), dan f1 per kelas
    precisions = results.box.p
    recalls = results.box.r
    f1_scores = results.box.f1

    print("=== F1 Score per Class ===")
    for i, f1 in enumerate(f1_scores):
        print(f"{class_names[i]} (class {i}):")
        print(f"  Precision = {precisions[i]:.4f}")
        print(f"  Recall    = {recalls[i]:.4f}")
        print(f"  F1 Score  = {f1:.4f}\n")


if __name__ == '__main__':
    eval()