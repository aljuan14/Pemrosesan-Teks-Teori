import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- KONFIGURASI ---
FILE_MANUAL = 'data_manual.csv'
FILE_FULL_RAW = 'analisis_sentimen_bersih.csv'
OUTPUT_FILE = 'analisis_sentimen_full_labeled.csv'


def train_and_predict():
    print("=== PROGRAM AUTO-LABELING DENGAN SVM ===")

    # 1. Load Data
    try:
        df_labeled = pd.read_csv(FILE_MANUAL)
        df_full = pd.read_csv(FILE_FULL_RAW)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"Data Training (Manual): {len(df_labeled)} baris")

    # 2. Preprocessing & Vectorization
    # Menggunakan TF-IDF dengan n-gram (1,2) agar frasa seperti "tidak suka" tertangkap satu kesatuan
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

    # Fit pada data yang sudah ada labelnya
    X = df_labeled['final_text'].fillna('')
    y = df_labeled['label']

    X_vectorized = tfidf.fit_transform(X)

    # 3. Training Model SVM
    # Kernel 'linear' adalah standar emas untuk klasifikasi teks
    model = SVC(kernel='linear', probability=True, random_state=42)

    # Cek Validasi (Cross Validation 5-fold)
    # Ini memberi gambaran akurasi modelmu saat ini
    scores = cross_val_score(model, X_vectorized, y, cv=5)
    print(f"\nEstimasi Akurasi Model: {scores.mean()*100:.2f}%")

    # Train pada SELURUH data manual
    model.fit(X_vectorized, y)
    print("Model berhasil dilatih!")

    # 4. Prediksi Sisa Data (Unlabeled)
    # Kita harus memisahkan mana yang sudah dilabel manual, mana yang belum
    # Membuat unique key untuk membedakan baris (text + subjek)
    df_labeled['temp_key'] = df_labeled['final_text'] + df_labeled['subjek']
    df_full['temp_key'] = df_full['final_text'] + df_full['subjek']

    # Filter: Ambil data di df_full yang KEY-nya TIDAK ada di df_labeled
    labeled_keys = set(df_labeled['temp_key'])
    df_unlabeled = df_full[~df_full['temp_key'].isin(labeled_keys)].copy()

    print(f"Sisa data yang akan dilabeli otomatis: {len(df_unlabeled)} baris")

    if len(df_unlabeled) > 0:
        # Transform data baru menggunakan TF-IDF yang sdh dilatih (JANGAN di-fit ulang)
        X_unlabeled = tfidf.transform(df_unlabeled['final_text'].fillna(''))

        # Prediksi Label
        predictions = model.predict(X_unlabeled)

        # Masukkan hasil prediksi
        df_unlabeled['label'] = predictions
        df_unlabeled['status'] = 'prediksi_model'

    # Beri tanda pada data manual
    df_labeled['status'] = 'manual'

    # 5. Gabungkan dan Simpan
    cols_to_keep = ['subjek', 'final_text', 'label', 'status']
    df_final = pd.concat(
        [df_labeled[cols_to_keep], df_unlabeled[cols_to_keep]], ignore_index=True)

    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSUKSES! Data tersimpan di '{OUTPUT_FILE}'")
    print(f"Total Data Akhir: {len(df_final)}")
    print("-" * 30)
    print("Contoh hasil prediksi model:")
    print(df_unlabeled[['final_text', 'label']].head(5))

    # Optional: Tampilkan laporan klasifikasi detail dari split test kecil
    print("\n--- Detail Performa Model (pada 20% data test split) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42)
    model_eval = SVC(kernel='linear')
    model_eval.fit(X_train, y_train)
    y_pred_eval = model_eval.predict(X_test)
    print(classification_report(y_test, y_pred_eval))


if __name__ == "__main__":
    train_and_predict()
