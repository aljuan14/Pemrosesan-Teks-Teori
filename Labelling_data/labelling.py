import pandas as pd
import os
import sys

# --- KONFIGURASI ---
INPUT_FILE = 'analisis_sentimen_bersih.csv'
OUTPUT_FILE = 'hasil_sentimen_labeled_per_tokoh.csv'

# Target per label UNTUK SETIAP TOKOH
# Jika target total 100, berarti per tokoh 50.
TARGET_PER_TOKOH = {
    'positif': 50,
    'negatif': 50,
    'netral': 25
}

# Mapping input keyboard ke label
LABEL_MAP = {
    '1': 'positif', 'p': 'positif', 'pos': 'positif',
    '2': 'negatif', 'n': 'negatif', 'neg': 'negatif',
    '3': 'netral', 'e': 'netral', 'net': 'netral'
}
# -------------------


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def main():
    print("=== PROGRAM PELABELAN NLP: STRATIFIED SAMPLING (PER TOKOH) ===")

    # 1. Load Data
    try:
        df_source = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"[ERROR] File '{INPUT_FILE}' tidak ditemukan.")
        return

    # Ambil daftar tokoh unik dari data
    subjects = df_source['subjek'].unique()
    print(f"Tokoh ditemukan: {', '.join(subjects)}")

    # Acak data di awal
    df_source = df_source.sample(
        frac=1, random_state=42).reset_index(drop=True)

    # 2. Siapkan/Baca File Output
    if os.path.exists(OUTPUT_FILE):
        try:
            df_done = pd.read_csv(OUTPUT_FILE)
            # Filter data source, buang yang sudah ada di output
            # Kita pakai kombinasi teks + subjek biar lebih unik
            df_done['key'] = df_done['final_text'] + df_done['subjek']
            df_source['key'] = df_source['final_text'] + df_source['subjek']

            labeled_keys = set(df_done['key'].tolist())
            df_work = df_source[~df_source['key'].isin(
                labeled_keys)].drop(columns=['key'])

            print(f"Melanjutkan sesi. {len(df_done)} data sudah terlabel.")
        except Exception as e:
            print(f"[ERROR] Gagal membaca file output: {e}")
            return
    else:
        # File baru
        pd.DataFrame(columns=['subjek', 'final_text', 'label']).to_csv(
            OUTPUT_FILE, index=False)
        df_done = pd.DataFrame(columns=['subjek', 'final_text', 'label'])
        df_work = df_source.copy()
        print("Memulai sesi baru.")

    # 3. Hitung Progress Saat Ini Per Tokoh
    # Struktur: counts[nama_tokoh][label] = jumlah
    counts = {s: {'positif': 0, 'negatif': 0, 'netral': 0} for s in subjects}

    if len(df_done) > 0:
        # Groupby subjek dan label untuk hitung jumlah
        progress = df_done.groupby(['subjek', 'label']).size()
        for s in subjects:
            for l in ['positif', 'negatif', 'netral']:
                if (s, l) in progress.index:
                    counts[s][l] = progress[(s, l)]

    # 4. Loop Pelabelan
    for index, row in df_work.iterrows():
        subjek = row['subjek']
        text = row['final_text']

        # Cek apakah TOKOH ini sudah memenuhi SEMUA kuota?
        is_subject_full = all(
            counts[subjek][l] >= TARGET_PER_TOKOH[l] for l in TARGET_PER_TOKOH)

        # Jika subjek ini sudah full semua kuotanya, otomatis skip tanpa tanya user
        if is_subject_full:
            continue

        # Cek apakah SEMUA tokoh sudah full? (Exit condition)
        all_full = True
        for s in subjects:
            if not all(counts[s][l] >= TARGET_PER_TOKOH[l] for l in TARGET_PER_TOKOH):
                all_full = False
                break

        if all_full:
            print("\n" + "="*50)
            print("SELAMAT! Semua target kuota untuk SEMUA tokoh sudah terpenuhi!")
            print("="*50)
            break

        # Tampilkan Status
        print("\n" + "="*60)
        print(f"STATUS KUOTA UNTUK: {subjek.upper()}")
        status_str = " | ".join(
            [f"{l.upper()}: {counts[subjek][l]}/{TARGET_PER_TOKOH[l]}" for l in TARGET_PER_TOKOH])
        print(f"[ {status_str} ]")

        # Tampilkan Status Tokoh Lain (sebagai info saja)
        other_subjects = [s for s in subjects if s != subjek]
        if other_subjects:
            print("-" * 20)
            for os_name in other_subjects:
                os_str = " | ".join(
                    [f"{l[0].upper()}:{counts[os_name][l]}" for l in TARGET_PER_TOKOH])
                print(f"Info {os_name}: {os_str}")
        print("="*60)

        print(f"TEKS:\n\"{text}\"")
        print("-" * 60)

        # Input Loop
        valid_input = False
        while not valid_input:
            prompt = f"Label untuk {subjek} (p/n/e) [s=Skip, q=Quit]: "
            user_input = input(prompt).lower().strip()

            if user_input in ['q', 'quit', 'exit']:
                print("Keluar...")
                sys.exit()

            if user_input in ['s', 'skip']:
                print(">> Skip.")
                valid_input = True
                continue

            if user_input in LABEL_MAP:
                label = LABEL_MAP[user_input]

                # Cek kuota spesifik
                if counts[subjek][label] >= TARGET_PER_TOKOH[label]:
                    print(
                        f"!! Kuota {label.upper()} untuk {subjek} SUDAH PENUH ({TARGET_PER_TOKOH[label]}). Pilih label lain atau Skip.")
                    continue
                else:
                    # Save
                    new_row = pd.DataFrame(
                        {'subjek': [subjek], 'final_text': [text], 'label': [label]})
                    new_row.to_csv(OUTPUT_FILE, mode='a',
                                   header=False, index=False)

                    counts[subjek][label] += 1
                    print(f">> Disimpan: {subjek} -> {label}")
                    valid_input = True
            else:
                print("!! Input salah. Gunakan p (positif), n (negatif), e (netral).")

    if len(df_work) == 0:
        print("Data sumber habis!")


if __name__ == "__main__":
    main()
