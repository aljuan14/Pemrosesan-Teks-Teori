import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- KONFIGURASI FILE ---
# Pastikan nama file ini sesuai dengan yang ada di folder kamu
FILE_MANUAL = 'dataset_labeling_ai.csv'
FILE_FULL_RAW = 'analisis_sentimen_bersih.csv'
OUTPUT_FILE = 'dataset_self_train_ai.csv'


def train_and_predict_pro():
    print("=== PROGRAM AUTO-LABELING PRO (ANTI-BIAS) ===")

    # 1. Load Data
    try:
        df_train = pd.read_csv(FILE_MANUAL)
        df_raw = pd.read_csv(FILE_FULL_RAW)
        print(f"[OK] Data Training Loaded: {len(df_train)} baris")
        print(f"[OK] Data Raw Loaded: {len(df_raw)} baris")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Preprocessing & Vectorization
    print("\n[Proses] Melakukan Vektorisasi Teks...")
    # ngram_range=(1,2) menangkap konteks frasa (misal: "tidak bagus")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

    # Fit pada data training manual
    X_train_text = df_train['final_text'].fillna('')
    y_train = df_train['label']

    X_train_vec = tfidf.fit_transform(X_train_text)

    # 3. Evaluasi Model (Untuk Laporan Bab 4)
    print("\n[Evaluasi] Mengecek Performa Model...")
    # Split 80:20 untuk tes internal
    X_tr, X_ts, y_tr, y_ts = train_test_split(
        X_train_vec, y_train, test_size=0.2, random_state=42, stratify=y_train)

    model_eval = SVC(kernel='linear', probability=True, random_state=42)
    model_eval.fit(X_tr, y_tr)
    y_pred_eval = model_eval.predict(X_ts)

    # Tampilkan Metrik Keren
    acc = accuracy_score(y_ts, y_pred_eval)
    print(f"\n>> AKURASI MODEL: {acc*100:.2f}%")
    print("-" * 60)
    print(classification_report(y_ts, y_pred_eval))
    print("-" * 60)
    print("Confusion Matrix (Tabel Kebenaran):")
    print(confusion_matrix(y_ts, y_pred_eval,
          labels=['positif', 'negatif', 'netral']))
    print("-" * 60)

    # 4. Training Full Model
    print("\n[Training] Melatih model final dengan 100% data manual...")
    model_final = SVC(kernel='linear', probability=True, random_state=42)
    model_final.fit(X_train_vec, y_train)

    # 5. Anti-Data Leakage (Pemisahan Data)
    print("[Filter] Memisahkan data yang belum dilabeli...")

    # Buat kunci unik (Text + Subjek)
    df_train['key'] = df_train['final_text'] + df_train['subjek']
    df_raw['key'] = df_raw['final_text'] + df_raw['subjek']

    # Ambil hanya data di RAW yang kuncinya TIDAK ADA di TRAINING
    labeled_keys = set(df_train['key'])
    df_unlabeled = df_raw[~df_raw['key'].isin(labeled_keys)].copy()

    print(f">> Data Sisa (Unlabeled): {len(df_unlabeled)} baris")

    # 6. Prediksi Sisa Data
    if len(df_unlabeled) > 0:
        print("[Prediksi] Sedang melabeli data sisa...")
        X_unlabeled = tfidf.transform(df_unlabeled['final_text'].fillna(''))

        # Prediksi Label & Confidence Score
        preds = model_final.predict(X_unlabeled)
        probs = model_final.predict_proba(X_unlabeled)
        # Ambil nilai probabilitas tertinggi
        confidence = np.max(probs, axis=1)

        df_unlabeled['label'] = preds
        df_unlabeled['status'] = 'prediksi_model'
        # Fitur baru: seberapa yakin si model?
        df_unlabeled['confidence'] = confidence
    else:
        print("Semua data sudah terlabeli di file manual!")

    # 7. Gabungkan & Simpan
    df_train['status'] = 'manual'
    df_train['confidence'] = 1.0  # Manual pasti yakin 100%

    cols_to_keep = ['subjek', 'final_text', 'label', 'status', 'confidence']
    df_final = pd.concat(
        [df_train[cols_to_keep], df_unlabeled[cols_to_keep]], ignore_index=True)

    df_final.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "="*50)
    print(f"SUKSES! File akhir disimpan: {OUTPUT_FILE}")
    print(f"Total Data: {len(df_final)}")
    print("="*50)

    # Intip hasil
    print("\nContoh Hasil Prediksi (dengan Confidence):")
    print(df_unlabeled[['final_text', 'label', 'confidence']].head())


if __name__ == "__main__":
    train_and_predict_pro()
