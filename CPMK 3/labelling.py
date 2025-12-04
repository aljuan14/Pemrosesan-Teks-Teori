import pandas as pd
import os
import sys

# --- KONFIGURASI ---
INPUT_FILE = 'analisis_sentimen_bersih.csv'  # Nama file input kamu
OUTPUT_FILE = 'hasil_sentimen_labeled.csv'   # Nama file output
TARGETS = {
    'positif': 100,
    'negatif': 100,
    'netral': 50
}
# -------------------


def clear_screen():
    # Membersihkan layar agar tampilan rapi
    os.system('cls' if os.name == 'nt' else 'clear')


def main():
    print("=== PROGRAM PELABELAN DATA NLP MANUAL ===")

    # 1. Load Data Input
    try:
        df_source = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(
            f"[ERROR] File '{INPUT_FILE}' tidak ditemukan. Pastikan file ada di folder yang sama.")
        return

    # Acak data agar tidak bias urutan (PENTING untuk NLP)
    df_source = df_source.sample(
        frac=1, random_state=42).reset_index(drop=True)

    # 2. Cek File Output & Hitung Progres
    if os.path.exists(OUTPUT_FILE):
        try:
            df_done = pd.read_csv(OUTPUT_FILE)
            # Hitung jumlah label yang sudah ada
            current_counts = df_done['label'].value_counts().to_dict()

            # Ambil daftar teks yang sudah dilabeli agar tidak muncul lagi
            # (Asumsi: kolom 'final_text' cukup unik untuk jadi identifier)
            labeled_texts = set(df_done['final_text'].tolist())

            # Filter df_source, buang yang sudah ada di labeled_texts
            df_work = df_source[~df_source['final_text'].isin(labeled_texts)]
            print(
                f"Melanjutkan sesi sebelumnya. {len(df_done)} data sudah terlabel.")
        except Exception as e:
            print(f"[ERROR] Gagal membaca file output: {e}")
            return
    else:
        # Jika file output belum ada, buat baru dengan Header
        pd.DataFrame(columns=['subjek', 'final_text', 'label']).to_csv(
            OUTPUT_FILE, index=False)
        current_counts = {}
        labeled_texts = set()
        df_work = df_source.copy()
        print("Memulai sesi baru.")

    # Pastikan semua key target ada di counter (set 0 jika belum ada)
    for key in TARGETS:
        if key not in current_counts:
            current_counts[key] = 0

    # 3. Loop Utama Pelabelan
    total_needed = sum(TARGETS.values())

    for index, row in df_work.iterrows():
        # Cek apakah semua target sudah terpenuhi
        if all(current_counts[k] >= TARGETS[k] for k in TARGETS):
            print("\n" + "="*40)
            print("SELAMAT! Semua target kuota data telah terpenuhi!")
            print(
                f"Positif: {current_counts.get('positif',0)}, Negatif: {current_counts.get('negatif',0)}, Netral: {current_counts.get('netral',0)}")
            print("="*40)
            break

        # Tampilkan Interface
        print("\n" + "-"*50)
        status_msg = " | ".join(
            [f"{k.upper()}: {current_counts.get(k,0)}/{TARGETS[k]}" for k in TARGETS])
        print(f"PROGRESS: [ {status_msg} ]")
        print("-"*50)

        print(f"SUBJEK : {row['subjek']}")
        print(f"TEKS   : \n\"{row['final_text']}\"")
        print("-"*50)

        # Loop validasi input user
        valid_input = False
        while not valid_input:
            user_input = input(
                "Label (1/p=Pos, 2/n=Neg, 3/e=Net, s=Skip, q=Quit): ").lower().strip()

            label = None
            if user_input in ['q', 'quit', 'exit']:
                print("Menyimpan dan keluar...")
                sys.exit()

            elif user_input in ['s', 'skip']:
                print(">> Data dilewati (Skip).")
                valid_input = True  # Lanjut ke row berikutnya tanpa save

            elif user_input in ['1', 'p', 'pos']:
                label = 'positif'
            elif user_input in ['2', 'n', 'neg']:
                label = 'negatif'
            elif user_input in ['3', 'e', 'net', 'netral']:
                label = 'netral'
            else:
                print("!! Input tidak valid. Gunakan: p, n, e, s, atau q.")
                continue

            if label:
                # Cek apakah kuota label tersebut sudah penuh
                if current_counts[label] >= TARGETS[label]:
                    print(
                        f"!! Kuota {label.upper()} sudah penuh ({TARGETS[label]}). Harap cari data jenis lain atau skip.")
                    # Kita loop lagi minta input ulang untuk data yang sama
                    continue
                else:
                    # SIMPAN DATA
                    new_row = pd.DataFrame({
                        'subjek': [row['subjek']],
                        'final_text': [row['final_text']],
                        'label': [label]
                    })
                    # Append mode, header=False karena file sudah dibuat di awal
                    new_row.to_csv(OUTPUT_FILE, mode='a',
                                   header=False, index=False)

                    # Update counter
                    current_counts[label] += 1
                    print(f">> Tersimpan sebagai {label.upper()}.")
                    valid_input = True

    if len(df_work) == 0:
        print("Data sumber sudah habis!")


if __name__ == "__main__":
    main()
