# 🚗 Vehicle Counting & Classification using YOLO

A final-year thesis project implementing **YOLOv8** for real-time vehicle **detection, tracking, counting, and classification** using a simple and intuitive **Tkinter GUI**.

---

## 🔧 Development Setup
- **Language:** Python 3.9  
- **Frameworks:** PyTorch, Tkinter  
- **Model:** YOLOv8 + ByteTrack  
- **Operating System:** Windows 11  
- **IDE:** VS Code  

---

## ⚙️ Features
- ROI-based vehicle counting  
- Real-time detection & tracking  
- Multi-class classification (car, motorbike, truck)  
- Direction-aware counting  
- Export results to Excel  
- User-friendly Tkinter interface  

---

## 🚀 Usage Steps

### **Step 1 — Upload Video**
Click the **"Upload Video"** button to select the video you want to analyze.  

---

### **Step 2 — Set ROI Line**
After the video loads:

    1. Click **two points** on the video frame to draw the ROI line.  
    2. **First click:** one side of the road  
    3. **Second click:** the opposite side  
    4. A **line** will appear—this is the counting boundary.

---

### **Step 3 — Configure Road Information**
Fill in the required fields:

- **Road Name** → Name of the street/area being analyzed  
- **Direction** → Example: *Kiri*, *kanan*, *Atas*, *Bawah*  

---

### **Step 4 — Start Analysis**

- Real-time YOLOv8 detection  
- ByteTrack multi-object tracking  
- Automatic classification (car, motorbike, truck, Bus)  
- Counter updates in real-time  

Bounding boxes, labels, and vehicle counts will be shown on the live video preview.

---

### **Step 5 — Export Results**
Click **"Export to Excel"** to save your vehicle counting data.

The exported report includes:

- Vehicle counts by category  
- Timestamp logs  
- Road name & traffic direction  
- Total summary of all counts  

---

## 📊 Output Overview
- Live video display with YOLO detection overlays  
- Counters for each vehicle class:
  - Car  
  - Motorbike  
  - Truck  
- Visual ROI line for verification  
- Excel report containing complete analysis data  

---

## 📦 Installation
```bash
git clone https://github.com/P11K/Klasifikasi-Kendaraan-Skripsi.git
cd Klasifikasi-Kendaraan-Skripsi
pip install -r requirements.txt
python main.py
