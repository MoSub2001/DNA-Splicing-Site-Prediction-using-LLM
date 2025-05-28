import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load k-mer log-probabilities from a single global CSV
# ─────────────────────────────────────────────────────────────────────────────

def load_kmer_probability_map(kmer_prob_csv):
    """
    Load a map: (distance, kmer) → ln(prob)
    """
    print(f"📥 Loading k-mer probabilities from: {kmer_prob_csv}")
    df = pd.read_csv(kmer_prob_csv)
    lnP_map = {}

    for _, row in df.iterrows():
        kmer = row['kmer']
        dist = int(row['distance'])
        prob = float(row['probability'])
        ln_p = np.log(prob + 1e-10)  # safe log to avoid -inf
        lnP_map[(dist, kmer)] = lnP_map.get((dist, kmer), 0.0) + ln_p  # sum if multiple entries

    print(f"✅ Loaded {len(lnP_map)} (distance, k-mer) entries.")
    return lnP_map

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build one lnP vector for a single sequence (shape = [504])
# ─────────────────────────────────────────────────────────────────────────────

def build_lnP_vector_for_sequence(sequence, lnP_map, max_back=200, max_forward=100, kmer_length=6):
    """
    Given a DNA sequence of length 504, compute log-probability sums per index.
    """
    seq_len = len(sequence)
    lnP_vector = np.zeros(seq_len, dtype=np.float32)

    for i in range(seq_len):
        ln_total = 0.0
        for dist in range(-max_back, max_forward + 1):
            start = i - dist
            if 0 <= start <= seq_len - kmer_length:
                kmer = sequence[start : start + kmer_length]
                ln_total += lnP_map.get((dist, kmer), 0.0)
        lnP_vector[i] = ln_total

    return torch.tensor(lnP_vector, dtype=torch.float32)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Process all chromosomes and sequences, and save outputs
# ─────────────────────────────────────────────────────────────────────────────

def process_all_sequences(
    data_dir="../chromosomeData",
    prob_file="../kmer_probabilities_2.csv",
    out_dir="predictions"
):
    os.makedirs(out_dir, exist_ok=True)
    lnP_map = load_kmer_probability_map(prob_file)

    for chrom in range(1, 23):
        csv_path = os.path.join(data_dir, f"ch{chrom}.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️  Missing file: {csv_path}, skipping.")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue

        if 'Token' not in df.columns:
            print(f" 'Token' column not found in {csv_path}, skipping.")
            continue

        print(f"🔍 Processing {csv_path} with {len(df)} sequences...")
        lnP_tensors = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"ch{chrom}"):
            token = row['Token']
            if isinstance(token, str) and len(token) == 504:
                lnP_tensor = build_lnP_vector_for_sequence(token, lnP_map)
                lnP_tensors.append(lnP_tensor)

        if lnP_tensors:
            stacked = torch.stack(lnP_tensors)  # shape: (N, 504)
            save_path = os.path.join(out_dir, f"probvec_ch{chrom}.pt")
            torch.save(stacked, save_path)
            print(f"✅ Saved {len(lnP_tensors)} sequences to {save_path}")
        else:
            print(f"⚠️  No valid sequences in {csv_path}")

    print(f"\n Done! All probability vectors saved to: {out_dir}/")

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    process_all_sequences()
