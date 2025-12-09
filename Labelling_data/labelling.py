import pandas as pd
import os
import sys

# ==========================================
# KONFIGURASI (SUDAH DISESUAIKAN)
# ==========================================
INPUT_FILE = 'dataset_tugas_purbaya_vs_srimulyani.csv'  # Nama file input
OUTPUT_FILE = 'hasil_labelling_manual.csv'              # Nama file hasil
# File untuk menyimpan data yang di-skip
SKIP_FILE = 'riwayat_skip.csv'

# PENTING: Nama kolom disesuaikan dengan dataset kamu
KOLOM_TEXT = 'text'     # Di CSV kamu kolomnya bernama 'text'
KOLOM_SUBJEK = 'tokoh'  # Di CSV kamu kolomnya bernama 'tokoh'

# Target hanya sebagai indikator (Boleh lebih/Over quota)
TARGET_PER_TOKOH = {
    'positif': 50,
    'negatif': 50,
    'netral': 50
}

# Mapping input keyboard
LABEL_MAP = {
    '1': 'positif', 'p': 'positif', 'pos': 'positif',
    '2': 'negatif', 'n': 'negatif', 'neg': 'negatif',
    '3': 'netral', 'e': 'netral', 'net': 'netral'
}

# ==========================================
# FUNGSI BANTUAN
# ==========================================


def clear_screen():
    # Membersihkan layar terminal (Windows/Mac/Linux)
    os.system('cls' if os.name == 'nt' else 'clear')


def load_data(filepath):
    """Membaca CSV dengan handling encoding otomatis"""
    try:
        return pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            return pd.read_csv(filepath, encoding='latin1')
        except:
            return pd.read_csv(filepath, encoding='ISO-8859-1')


def main():
    clear_screen()
    print("=== PROGRAM PELABELAN: AUTO-RESUME & OVER QUOTA ===")

    # 1. Cek File Input
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] File '{INPUT_FILE}' tidak ditemukan.")
        print("Pastikan file CSV berada di folder yang sama dengan script ini.")
        return

    df_source = load_data(INPUT_FILE)

    # Validasi Kolom
    if KOLOM_TEXT not in df_source.columns or KOLOM_SUBJEK not in df_source.columns:
        print(f"[ERROR] Nama kolom tidak cocok!")
        print(f"Dicari di script : '{KOLOM_TEXT}' dan '{KOLOM_SUBJEK}'")
        print(f"Yang ada di CSV  : {df_source.columns.tolist()}")
        return

    # 2. Load File Output (Data yg sudah selesai dilabel)
    labeled_keys = set()
    if os.path.exists(OUTPUT_FILE):
        df_done = load_data(OUTPUT_FILE)
        # Kunci unik = Text + Tokoh (agar tidak duplikat)
        labeled_keys = set((df_done[KOLOM_TEXT].astype(
            str) + df_done[KOLOM_SUBJEK].astype(str)).tolist())
        print(f"> Data Sudah Terlabel : {len(df_done)}")
    else:
        # Buat file output baru
        pd.DataFrame(columns=[KOLOM_SUBJEK, KOLOM_TEXT, 'label']).to_csv(
            OUTPUT_FILE, index=False)
        df_done = pd.DataFrame(columns=[KOLOM_SUBJEK, KOLOM_TEXT, 'label'])
        print("> Data Sudah Terlabel : 0 (Mulai Baru)")

    # 3. Load File Skip (Data yg pernah di-skip user agar tidak muncul lagi)
    skipped_keys = set()
    if os.path.exists(SKIP_FILE):
        df_skip = load_data(SKIP_FILE)
        skipped_keys = set((df_skip[KOLOM_TEXT].astype(
            str) + df_skip[KOLOM_SUBJEK].astype(str)).tolist())
        print(f"> Data Pernah Di-Skip : {len(df_skip)}")
    else:
        pd.DataFrame(columns=[KOLOM_SUBJEK, KOLOM_TEXT]
                     ).to_csv(SKIP_FILE, index=False)
        print("> Data Pernah Di-Skip : 0")

    # 4. Filter Data Source
    # Buang data yang sudah ada di Output ATAU di Skip
    df_source['key'] = df_source[KOLOM_TEXT].astype(
        str) + df_source[KOLOM_SUBJEK].astype(str)

    exclude_keys = labeled_keys.union(skipped_keys)
    df_work = df_source[~df_source['key'].isin(
        exclude_keys)].drop(columns=['key'])

    # Acak urutan (Random State 42 agar urutan konsisten saat dilanjutkan)
    df_work = df_work.sample(frac=1, random_state=42).reset_index(drop=True)

    # Hitung progress statistik saat ini
    subjects = df_source[KOLOM_SUBJEK].unique()
    counts = {s: {'positif': 0, 'negatif': 0, 'netral': 0} for s in subjects}

    if len(df_done) > 0:
        # Hitung jumlah label per tokoh
        progress = df_done.groupby([KOLOM_SUBJEK, 'label']).size()
        for s in subjects:
            for l in ['positif', 'negatif', 'netral']:
                if (s, l) in progress.index:
                    counts[s][l] = progress[(s, l)]

    print(f"\nSisa antrian data: {len(df_work)} baris.")
    input("Tekan Enter untuk MELANJUTKAN sesi terakhir...")

    # 5. Loop Pelabelan Utama
    for index, row in df_work.iterrows():
        clear_screen()
        subjek = row[KOLOM_SUBJEK]
        text = row[KOLOM_TEXT]

        # --- TAMPILAN STATUS ---
        print(f"TOKOH: {str(subjek).upper()}")
        print("-" * 60)

        status_parts = []
        for l in ['positif', 'negatif', 'netral']:
            curr = counts[subjek][l]
            tgt = TARGET_PER_TOKOH[l]

            # Indikator jika sudah memenuhi/melebihi target
            if curr >= tgt:
                surplus = curr - tgt
                # Info surplus
                status_str = f"{l.upper()}: {curr}/{tgt} (+{surplus})"
            else:
                status_str = f"{l.upper()}: {curr}/{tgt}"

            status_parts.append(status_str)

        print(" | ".join(status_parts))
        print("="*60)
        # Menampilkan teks
        print(f"TEKS:\n\n\"{text}\"\n")
        print("="*60)

        print(
            "CONTROLS: [p]=Positif | [n]=Negatif | [e]=Netral | [s]=Skip | [q]=Quit")

        # --- INPUT LOOP ---
        valid_input = False
        while not valid_input:
            user_input = input("Label > ").lower().strip()

            # Opsi QUIT (Keluar tanpa simpan current data, agar muncul lagi nanti)
            if user_input in ['q', 'quit', 'exit']:
                print("Keluar...")
                sys.exit()

            # Opsi SKIP (Simpan ke file skip, tidak akan muncul lagi)
            if user_input in ['s', 'skip']:
                new_skip = pd.DataFrame({
                    KOLOM_SUBJEK: [subjek],
                    KOLOM_TEXT: [text]
                })
                # Simpan mode append
                new_skip.to_csv(SKIP_FILE, mode='a', header=False, index=False)

                print(">> Di-Skip (Disimpan ke riwayat skip).")
                valid_input = True
                continue

            # Opsi LABEL (Simpan ke hasil)
            if user_input in LABEL_MAP:
                label = LABEL_MAP[user_input]

                new_row = pd.DataFrame({
                    KOLOM_SUBJEK: [subjek],
                    KOLOM_TEXT: [text],
                    'label': [label]
                })
                # Simpan mode append
                new_row.to_csv(OUTPUT_FILE, mode='a',
                               header=False, index=False)

                counts[subjek][label] += 1

                print(f">> Disimpan: {subjek} -> {label}")
                valid_input = True
            else:
                print("!! Input salah. Gunakan p, n, e, s, atau q.")

    print("Data sumber habis! (Semua data telah dilabeli atau di-skip)")


if __name__ == "__main__":
    main()
