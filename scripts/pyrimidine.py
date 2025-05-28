#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

def compute_pyrimidine_content(seq: str, window: int = 20) -> np.ndarray:
    seq = seq.upper()
    n = len(seq)
    is_pyr = np.fromiter((1 if b in ("C","T") else 0 for b in seq), int, n)
    csum = np.concatenate([[0], np.cumsum(is_pyr)])
    arr = np.zeros(n, float)
    for i in range(n):
        start = max(0, i - window + 1)
        total = csum[i+1] - csum[start]
        length = i - start + 1
        arr[i] = total / length
    return arr

def process_first_three_chromosomes(csv_dir: str, output_dir: str, window: int = 20):
    os.makedirs(output_dir, exist_ok=True)
    # pick first three CSV files
    csvs = sorted(f for f in os.listdir(csv_dir) if f.endswith('.csv'))[:3]
    for fn in csvs:
        chrom = os.path.splitext(fn)[0]
        print(f"[+] Processing {chrom}")
        df = pd.read_csv(os.path.join(csv_dir, fn))
        seqs = df['Token'].astype(str).tolist()

        # compute pyr-content for each sequence
        pyr_list = [compute_pyrimidine_content(seq, window) for seq in seqs]

        # find the maximum length
        max_len = max(arr.shape[0] for arr in pyr_list)

        # pad each to max_len with nan
        padded = np.full((len(pyr_list), max_len), np.nan, dtype=float)
        for i, arr in enumerate(pyr_list):
            padded[i, :arr.shape[0]] = arr

        # save a single 2D array
        out_file = os.path.join(output_dir, f"{chrom}_pyrimidine.npy")
        np.save(out_file, padded)
        print(f"    → saved {out_file} (shape {padded.shape})")

if __name__ == "__main__":
    CSV_DIR    = "../chromosomeData"
    OUTPUT_DIR = "./pyrimidine_cache"
    WINDOW     = 20

    process_first_three_chromosomes(CSV_DIR, OUTPUT_DIR, WINDOW)
