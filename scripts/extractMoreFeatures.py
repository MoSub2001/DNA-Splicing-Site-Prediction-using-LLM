#!/usr/bin/env python3
"""
extractMoreFeatures_strict.py

Two‐pass SLURM‐friendly script to:
  1) compute a global 95th‐percentile signal cutoff across all your
     504‐nt window CSVs, then
  2) re‐export each CSV with two new features:
       - H3K36me3_peak_strict  (per‐base mask at that cutoff)
       - has_strong_H3K36me3   (any‐base‐above‐cutoff flag)
"""

import glob
import os
import re
import pandas as pd
import numpy as np
import pyBigWig

# ── CONFIG ──────────────────────────────────────────────────
CSV_PATTERN  = "../newChrData/ch*_with_coords_with_H3K36me3.csv"
BIGWIG_PATH  = "ENCFF601VTB.bigWig"
PERCENTILE   = 95  # choose top 5%
# ── END CONFIG ───────────────────────────────────────────────

def normalize_chrom(c, valid_chroms):
    c = str(c)
    if c in valid_chroms:
        return c
    stripped = re.sub(r'^chr|^ch', '', c)
    cand1 = 'chr' + stripped
    if cand1 in valid_chroms:
        return cand1
    cand2 = 'chr' + c
    if cand2 in valid_chroms:
        return cand2
    return c

def fetch_signal(bw, chrom, start, end):
    return np.nan_to_num(
        bw.values(chrom, int(start), int(end), numpy=True),
        nan=0.0
    )

def safe_fetch(bw, chrom, start, end, length):
    try:
        return fetch_signal(bw, chrom, start, end)
    except RuntimeError:
        print(f"⚠️ Skipping invalid interval {chrom}:{start}-{end}")
        return np.zeros(length, dtype=float)

def clamp_and_filter(df, valid_chroms):
    # normalize
    df['chrom_norm'] = df['chrom'].apply(lambda c: normalize_chrom(c, valid_chroms))
    df = df[df['chrom_norm'].isin(valid_chroms)].reset_index(drop=True)

    # clamp coords
    df['start'] = df['start'].astype(int).clip(lower=0)
    df['end']   = df['end'].astype(int)
    df['end']   = df.apply(
        lambda r: min(r['end'], valid_chroms[r['chrom_norm']]),
        axis=1
    )
    df = df[df['start'] < df['end']].reset_index(drop=True)
    return df

if __name__ == "__main__":
    print(f"Opening BigWig: {BIGWIG_PATH}")
    bw = pyBigWig.open(BIGWIG_PATH)
    valid_chroms = bw.chroms()

    files = sorted(glob.glob(CSV_PATTERN))
    if not files:
        raise FileNotFoundError(f"No files match {CSV_PATTERN}")

    # ── PASS 1: collect all signals ─────────────────────────────
    print("Pass 1: gathering signal arrays to compute global cutoff…")
    all_sigs = []
    for f in files:
        df = pd.read_csv(f)
        df = clamp_and_filter(df, valid_chroms)
        for _, r in df.iterrows():
            length = int(r['end']) - int(r['start'])
            arr = safe_fetch(bw, r['chrom_norm'], r['start'], r['end'], length)
            all_sigs.append(arr)

    all_vals   = np.concatenate(all_sigs)
    cutoff_val = np.percentile(all_vals, PERCENTILE)
    print(f"Global {PERCENTILE}th percentile cutoff = {cutoff_val:.5f}")

    # ── PASS 2: re‐process and write out with new features ───────
    print("Pass 2: thresholding each window and writing new CSVs…")
    for f in files:
        print(f" → {f}")
        df = pd.read_csv(f)
        df = clamp_and_filter(df, valid_chroms)

        sigs = []
        peaks_strict = []
        has_peak = []
        for _, r in df.iterrows():
            length = int(r['end']) - int(r['start'])
            arr = safe_fetch(bw, r['chrom_norm'], r['start'], r['end'], length)
            sigs.append(arr)
            mask = (arr > cutoff_val).astype(int)
            peaks_strict.append(mask)
            has_peak.append(int(mask.any()))

        df['H3K36me3_sig']           = sigs
        df['H3K36me3_peak_strict']   = peaks_strict
        df['has_strong_H3K36me3']    = has_peak

        out = os.path.splitext(f)[0] + "_with_H3K36me3.csv"
        df.drop(columns=['chrom_norm']).to_csv(out, index=False)
        print(f"    ➞ wrote {len(df)} rows to {out}")

    bw.close()
    print("All done.")
