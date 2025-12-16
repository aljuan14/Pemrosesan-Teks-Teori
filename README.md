Analisis Sentimen Komparatif Terhadap Kebijakan Fiskal: Era Baru Purbaya Yudhi Sadewa vs. Legacy Sri Mulyani Indrawati

Project ini disusun untuk memenuhi tugas mata kuliah **Pemrosesan Teks**, mencakup implementasi **CPMK 2 (Data Acquisition & Preprocessing)** dan **CPMK 3 (Sentiment Analysis Modeling)**.

Fokus utama project ini adalah menganalisis sentimen masyarakat (Positif, Negatif, Netral) dari komentar media sosial menggunakan pendekatan *Semi-Supervised Learning* untuk mengatasi keterbatasan data berlabel.

---

## 📌 Alur Pengerjaan (Pipeline)

Project ini dibagi menjadi dua tahap utama:

### 1️⃣ CPMK 2: Crawling & Preprocessing
Pada tahap ini, kami melakukan pengumpulan data mentah dan pembersihan data agar siap diolah.
* **Data Source:** Crawling komentar dari platform media sosial (YouTube/Berita Online) terkait topik ekonomi dan kinerja menteri.
* **Preprocessing Pipeline:**
    1.  *Case Folding* (Mengubah huruf menjadi kecil).
    2.  *Cleaning* (Menghapus angka, tanda baca, emoji, dan karakter aneh).
    3.  *Stopword Removal* (Menghapus kata hubung yang tidak bermakna).
    4.  *Normalization* (Memperbaiki *typo* atau singkatan).
* **Output:** File `analisis_sentimen_bersih.csv` (Data bersih namun belum memiliki label).

### 2️⃣ CPMK 3: Modeling & Auto-Labeling (Anti-Bias)
Pada tahap ini, kami membangun model klasifikasi sentimen. Karena melabeli ribuan data secara manual memakan waktu, kami menggunakan strategi *Semi-Supervised Learning*.

* **Manual Labeling:** Melabeli sebagian kecil data (Initial Seed) sebagai acuan model.
* **Experiment (Anti-Bias):** Kami menemukan tantangan berupa *Class Imbalance* di mana data dominan Positif/Negatif, sehingga model awal gagal mendeteksi sentimen Netral.
    * *Solusi:* Menambahkan 50 data sampel Netral secara spesifik ke dalam data latih.
* **Modeling:** Menggunakan algoritma Machine Learning (SVM/Naive Bayes) dengan fitur ekstraksi TF-IDF.

---

## 📊 Laporan Eksperimen & Hasil Evaluasi

Kami membandingkan dua skenario pelatihan untuk membuktikan pentingnya keseimbangan data kelas Netral.

| Metrik | Skenario A (Awal) | Skenario B (Final/Anti-Bias) |
| :--- | :---: | :---: |
| **Jumlah Data Latih** | 250 Data | **300 Data** (Ada penambahan Netral) |
| **Akurasi Total** | 74.00% | 58.33% |
| **Kemampuan Deteksi Netral** | **0% (Gagal Total)** | **Naik Signifikan (F1: 0.53)** |
| **Kesimpulan Model** | *Overfitting* & Bias Ekstrem | Lebih *Robust* & Adil |

> **Catatan:** Meskipun akurasi Skenario A terlihat tinggi (74%), model tersebut **cacat** karena menganggap semua komentar Netral sebagai Positif/Negatif. Skenario B (58%) dipilih sebagai model terbaik karena mampu mengenali ketiga kelas sentimen dengan baik.

---

## 📂 Struktur File & Dataset

* `analisis_sentimen_bersih.csv` : Hasil output CPMK 2 (Data bersih hasil crawling).
* `data_manual.csv` : Data yang dilabeli manual oleh manusia (Ground Truth).
* `data_manual_updated.csv` : Data manual yang sudah diperkaya dengan sampel Netral (Digunakan untuk Skenario B).
* `dataset_self_train_updated.csv` : **Hasil Akhir**. Dataset penuh yang sudah dilabeli otomatis oleh AI berdasarkan model terbaik.

## 🚀 Cara Menjalankan Program

### Prasyarat
Install library yang dibutuhkan:
```bash
pip install pandas scikit-learn textblob
