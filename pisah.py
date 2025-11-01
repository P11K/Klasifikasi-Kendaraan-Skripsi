import cv2
import os
from tkinter import Tk, filedialog

def extract_n_frames(video_path, output_folder, total_frames_to_extract):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames_to_extract > total_frames:
        print(f"Video hanya memiliki {total_frames} frame, akan diekstrak semuanya.")
        total_frames_to_extract = total_frames

    interval = total_frames // total_frames_to_extract

    frame_idx = 0
    saved = 0
    while cap.isOpened() and saved < total_frames_to_extract:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            filename = os.path.join(output_folder, f"frame_{saved:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"Selesai: {saved} frame disimpan ke {output_folder}")

# ==== PILIH VIDEO DARI FILE MANAGER ====
def main():
    # Sembunyikan window Tkinter utama
    root = Tk()
    root.withdraw()

    # Buka dialog untuk memilih file video
    video_path = filedialog.askopenfilename(
        title="Pilih file video",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )

    if not video_path:
        print("Tidak ada file yang dipilih.")
        return

    output_folder = "Anotasi Malam/images"
    total_frames_to_extract = int(input("Masukkan jumlah frame yang ingin diekstrak: "))
    
    extract_n_frames(video_path, output_folder, total_frames_to_extract)

if __name__ == '__main__':
    main()
