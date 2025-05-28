import os
import ast
from collections import Counter
import pandas as pd
print("I am here")
# === CONFIGURATION ===
DATA_DIR = "chromosomeData"  # Change to your actual folder path
K = 6
BACK = 200
FORWARD = 100
OUTPUT_CSV = "kmer_position_probabilities.csv"
SEQ_LENGTH = 504

# === HELPER: Extract kmers and their position from splice site ===
def get_kmer_positions(seq, k=6):
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        rel_pos = i - BACK
        yield kmer, rel_pos

# === GLOBAL COUNTERS ===
kmer_pos_counts = Counter()
position_totals = Counter()

# === LIST CSV FILES ===
csv_files = sorted([
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if f.endswith(".csv") and os.path.getsize(os.path.join(DATA_DIR, f)) > 0
])

# === PROCESS FILES ONE AT A TIME ===
previous_row = None
for file_index, file_path in enumerate(csv_files):
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            print(f"⚠️ Skipping {file_path} (empty DataFrame)")
            continue
    except Exception as e:
        print(f" Failed to read {file_path}: {e}")
        continue

    df = df.reset_index(drop=True)

    for i in range(len(df)):
        row = df.iloc[i]
        exon_str = row.get('[exon Indices]', "")
        if not isinstance(exon_str, str) or exon_str.strip() in ("", "[]"):
            continue
        try:
            exons = ast.literal_eval(exon_str)
        except:
            continue

        sequence = row['Token'].upper()
        used_indices = set()

        # Look-ahead: next row
        next_row = None
        if i + 1 < len(df):
            next_row = df.iloc[i + 1]
        elif file_index + 1 < len(csv_files):
            try:
                next_df = pd.read_csv(csv_files[file_index + 1])
                if not next_df.empty:
                    next_row = next_df.iloc[0]
            except:
                pass

        for exon in exons:
            start = exon['start']
            end = exon['end']
            if start in used_indices:
                continue

            forward_len = min(FORWARD, end - start + 1)
            a1 = start + forward_len

            # === FORWARD PART ===
            forward_part = sequence[start:min(a1, len(sequence))]
            if a1 > len(sequence) and next_row is not None:
                next_exon_str = next_row.get('[exon Indices]', "")
                if isinstance(next_exon_str, str) and next_exon_str.strip() != "[]":
                    try:
                        next_exons = ast.literal_eval(next_exon_str)
                        for next_exon in next_exons:
                            if next_exon['start'] == 0:
                                needed = a1 - len(sequence)
                                forward_part += next_row['Token'].upper()[:needed]
                                used_indices.add(0)
                                break
                    except:
                        pass

            # === BACKWARD PART ===
            back_start = start - BACK
            if back_start >= 0:
                backward_part = sequence[back_start:start]
            else:
                need_from_prev = -back_start
                backward_part = ""
                if previous_row is not None:
                    prev_seq = previous_row['Token'].upper()
                    if need_from_prev <= len(prev_seq):
                        backward_part = prev_seq[-need_from_prev:] + sequence[:start]
                    else:
                        backward_part = prev_seq + sequence[:start]
                else:
                    backward_part = sequence[:start]

            # === COMBINE WINDOW AND PROCESS K-MERS ===
            window = backward_part + forward_part
            if len(window) >= K:
                for kmer, rel_pos in get_kmer_positions(window, K):
                    kmer_pos_counts[(kmer, rel_pos)] += 1
                    position_totals[rel_pos] += 1

        previous_row = row  # Save for look-back on next row

# === CONVERT TO PROBABILITIES AND SORT ===
records = []
for (kmer, pos), count in kmer_pos_counts.items():
    total_at_pos = position_totals[pos]
    prob = count / total_at_pos
    records.append((kmer, pos, prob))

# Create DataFrame and sort it
df_out = pd.DataFrame(records, columns=["kmer", "distance", "probability"])
df_out = df_out.sort_values(by=["distance", "kmer"]).reset_index(drop=True)
df_out.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Done! Saved sorted k-mer probabilities to: {OUTPUT_CSV}")
