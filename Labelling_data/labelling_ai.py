import pandas as pd
import numpy as np

# --- KONFIGURASI ---
FILE_INPUT = 'analisis_sentimen_bersih.csv'
FILE_OUTPUT = 'dataset_labeling_ai.csv'

# --- KAMUS SENTIMEN (LOGIKA AI) ---
# Saya menggunakan kata kunci yang kuat untuk menentukan sentimen
KATAKUNCI_POSITIF = [
    'mantap', 'keren', 'cerdas', 'hebat', 'bagus', 'setuju', 'dukung',
    'amanah', 'jujur', 'semangat', 'maju', 'sukses', 'terbaik', 'bangga',
    'solusi', 'berkah', 'lindungi', 'sayang', 'cinta', 'salut', 'profesional',
    'bijak', 'rapi', 'bersih', 'optimis', 'harapan', 'percaya', 'gas',
    'top', 'juara', 'membangun', 'karya', 'prestasi', 'jempol', 'akurat'
]

KATAKUNCI_NEGATIF = [
    'korupsi', 'utang', 'benci', 'gagal', 'mundur', 'bodoh', 'tolol',
    'sengsara', 'miskin', 'hancur', 'rusak', 'beban', 'sampah', 'bohong',
    'palsu', 'kecewa', 'marah', 'takut', 'ancam', 'jahat', 'licik',
    'pajak', 'naik', 'mahal', 'susah', 'maling', 'rampok', 'copet',
    'lengser', 'turun', 'kacau', 'gaduh', 'ribut', 'omong kosong', 'bual'
]


def hitung_sentimen(teks):
    if not isinstance(teks, str):
        return 0

    skor = 0
    teks_lower = teks.lower()

    # Tambah nilai jika ada kata positif
    for kata in KATAKUNCI_POSITIF:
        if kata in teks_lower:
            skor += 1

    # Kurangi nilai jika ada kata negatif
    for kata in KATAKUNCI_NEGATIF:
        if kata in teks_lower:
            skor -= 1

    return skor


def main():
    print("=== AI AUTO-LABELING (LEXICON BASED) ===")

    # 1. Load Data
    try:
        df = pd.read_csv(FILE_INPUT)
        print(f"Total Data Mentah: {len(df)}")
    except FileNotFoundError:
        print("File tidak ditemukan.")
        return

    # 2. Scoring Setiap Kalimat
    print("Sedang menilai sentimen...")
    df['ai_score'] = df['final_text'].apply(hitung_sentimen)

    # 3. Filtering Kuota (100 Pos, 100 Neg, 50 Net)

    # A. Ambil 100 POSITIF Terbaik (Skor Tertinggi)
    df_pos = df[df['ai_score'] > 0].sort_values(by='ai_score', ascending=False)
    if len(df_pos) >= 100:
        df_pos_final = df_pos.head(100).copy()
        df_pos_final['label'] = 'positif'
    else:
        print(
            f"Peringatan: Hanya ditemukan {len(df_pos)} data positif (kurang dari 100).")
        df_pos_final = df_pos.copy()
        df_pos_final['label'] = 'positif'

    # B. Ambil 100 NEGATIF Terbaik (Skor Terendah)
    df_neg = df[df['ai_score'] < 0].sort_values(by='ai_score', ascending=True)
    if len(df_neg) >= 100:
        df_neg_final = df_neg.head(100).copy()
        df_neg_final['label'] = 'negatif'
    else:
        print(
            f"Peringatan: Hanya ditemukan {len(df_neg)} data negatif (kurang dari 100).")
        df_neg_final = df_neg.copy()
        df_neg_final['label'] = 'negatif'

    # C. Ambil 50 NETRAL (Skor 0)
    # Kita acak biar variatif, karena biasanya data skor 0 banyak sekali
    df_net = df[df['ai_score'] == 0].sample(frac=1, random_state=42)
    if len(df_net) >= 50:
        df_net_final = df_net.head(50).copy()
        df_net_final['label'] = 'netral'
    else:
        df_net_final = df_net.copy()
        df_net_final['label'] = 'netral'

    # 4. Gabungkan
    df_result = pd.concat(
        [df_pos_final, df_neg_final, df_net_final], ignore_index=True)

    # Bersihkan kolom skor (biar rapi seperti manual)
    df_result = df_result[['subjek', 'final_text', 'label']]

    # Acak urutan akhir
    df_result = df_result.sample(
        frac=1, random_state=42).reset_index(drop=True)

    # 5. Simpan
    df_result.to_csv(FILE_OUTPUT, index=False)

    print("\n" + "="*50)
    print("LABELING AI SELESAI!")
    print(f"File tersimpan: {FILE_OUTPUT}")
    print("\nKomposisi Label AI:")
    print(df_result['label'].value_counts())
    print("="*50)


if __name__ == "__main__":
    main()
