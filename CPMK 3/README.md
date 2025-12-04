# Laporan Eksperimen: Auto-Labeling Analisis Sentimen (Anti-Bias)

Project ini bertujuan untuk melakukan pelabelan otomatis (*auto-labeling*) pada data sentimen publik terhadap tokoh politik menggunakan pendekatan *Machine Learning* (Semi-Supervised Learning).

Eksperimen ini membandingkan kinerja model sebelum dan sesudah penambahan data latih kelas **Netral**.

## 📊 Ringkasan Hasil Eksperimen

Kami melakukan dua kali pengujian pelatihan model:
1.  **Skenario A:** Menggunakan 250 data latih awal (dominan Positif/Negatif).
2.  **Skenario B:** Menggunakan 300 data latih (ditambah 50 data Netral untuk menyeimbangkan kelas).

### Tabel Perbandingan Performa

| Metrik | Skenario A (250 Data) | Skenario B (300 Data - Updated) |
| :--- | :---: | :---: |
| **Total Akurasi** | **74.00%** | 58.33% |
| **Precision (Netral)** | 0.00 (Gagal Total) | **0.48** |
| **Recall (Netral)** | 0.00 (Gagal Total) | **0.60** |
| **F1-Score (Netral)** | 0.00 | **0.53** |
| **Keseimbangan Kelas** | Sangat Bias (Positif/Negatif) | Lebih Seimbang (Anti-Bias) |

---

## 🧐 Analisis Mendalam

Meskipun **Skenario A** memiliki akurasi total yang lebih tinggi (74%), model tersebut **mengalami overfitting dan bias parah**. Model gagal mengenali sentimen netral sama sekali (F1-Score 0.00).

### 1. Masalah pada Skenario A (250 Data)
* **Error:** *UndefinedMetricWarning: Precision is ill-defined.*
* **Analisis:** Model ini memaksakan semua prediksi ke dalam kategori **Positif** atau **Negatif**.
* **Bukti:** Dari 10 data tes Netral, tidak ada satupun yang ditebak benar (0 benar).
* **Kesimpulan:** Model ini tidak valid untuk klasifikasi 3 kelas (Positif, Netral, Negatif).

### 2. Perbaikan pada Skenario B (300 Data - Dengan Tambahan Netral)
* **Peningkatan:** Meskipun akurasi global turun ke 58%, kemampuan model mengenali konteks **Netral** meningkat drastis.
* **Bukti:** F1-Score untuk kelas Netral naik dari **0.00 menjadi 0.53**.
* **Confusion Matrix:** Model kini mampu membedakan ketiga kelas dengan distribusi yang lebih wajar.
* **Kesimpulan:** Model ini jauh lebih *robust* dan representatif untuk data dunia nyata yang mengandung banyak opini netral/faktual.

---

## 📂 Struktur File Output

Hasil prediksi disimpan dalam dua file terpisah berdasarkan eksperimen:

1.  `dataset_self_train_ai.csv`
    * Hasil dari model Skenario A (Tanpa penambahan data netral).
    * Cenderung bias ke ekstrem (Sangat Positif/Sangat Negatif).
    
2.  `dataset_self_train_updated.csv` ✅ **(REKOMENDASI)**
    * Hasil dari model Skenario B (Dengan 50 data netral tambahan).
    * Mengandung prediksi label 'netral' yang lebih akurat.

---

## 🚀 Cara Menjalankan

Pastikan environment Python sudah aktif, lalu jalankan perintah berikut untuk melatih ulang model dengan data terbaru:

```bash
# Pastikan data_manual.csv sudah mencakup 300 data (250 lama + 50 baru)
python train_manual.py
