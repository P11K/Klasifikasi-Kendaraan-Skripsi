import matplotlib.pyplot as plt
import numpy as np

# Label kelas dan posisi
classes = ['Bus', 'Motor', 'Truck', 'Mobil']
x = np.arange(len(classes))
width = 0.35

# Data metrik
precision_siang = [0.979, 0.758, 0.928, 0.857]
recall_siang = [0.849, 0.806, 0.974, 0.789]
f1_siang = [0.8967, 0.7812, 0.9503, 0.8211]
map50_siang = [0.898, 0.821, 0.991, 0.860]

precision_malam = [0.0391, 0.866, 0.936, 0.723]
recall_malam = [0.0376, 0.773, 0.894, 0.747]
f1_malam = [0.0384, 0.8169, 0.9142, 0.7346]
map50_malam = [0.0229, 0.847, 0.954, 0.81]

# Dictionary metrik
metrics = {
    'Precision': (precision_siang, precision_malam),
    'Recall': (recall_siang, recall_malam),
    'F1-Score': (f1_siang, f1_malam),
    'mAP50': (map50_siang, map50_malam)
}

# Buat grafik
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, (metric_name, (data_siang, data_malam)) in enumerate(metrics.items()):
    axs[i].bar(x - width/2, data_siang, width, label='Siang', color='skyblue')
    axs[i].bar(x + width/2, data_malam, width, label='Malam', color='orange')
    axs[i].set_title(f'Perbandingan {metric_name} Siang dan Malam')
    axs[i].set_xticks(x)
    axs[i].set_xticklabels(classes)
    axs[i].set_ylim(0, 1.05)
    axs[i].legend()
    axs[i].set_ylabel(metric_name)

fig.suptitle('Perbandingan Semua Metrik Evaluasi (Siang dan Malam)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
