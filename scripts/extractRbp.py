#!/usr/bin/env python3
"""
extract_rbp_features_debug.py

Script to append RBP binding masks to your 504-nt window CSVs in-place,
with contig-normalization and debug output to catch mismatches.

Requirements:
  - Python 3
  - pandas, numpy
  - bedtools installed and in PATH

Usage:
  Just run:
    python extract_rbp_features_debug.py
"""

import glob
import os
import subprocess
import tempfile

import pandas as pd
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────

CSV_PATTERN = "../newChrData/ch*_with_coords_with_H3K36me3.csv"  # adjust to match your files

# Map RBP names to their original peak BED paths
RBP_PEAKS = {
    "PTBP1": "696_02.basedon_696_02.peaks.l2inputnormnew.bed.compressed.bed.narrowPeak.encode.bed",
    # add more RBPs as needed
}

# ── End Configuration ────────────────────────────────────────────────────────

def normalize_chrom_name(c):
    """Ensure chromosome names are 'chrN'."""
    c = str(c)
    if c.startswith("chr"):
        return c
    # strip leading 'ch' or 'CH' if present
    if c.lower().startswith("ch"):
        return "chr" + c[2:]
    return "chr" + c

def normalize_peak_bed(peaks_bed):
    """
    Create a temp BED with normalized 'chr' contigs.
    Returns (normalized_bed_path, set_of_contigs_seen)
    """
    norm = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".norm.bed")
    contigs = set()
    with open(peaks_bed) as inp:
        for line in inp:
            if line.startswith(("#","track")):
                continue
            parts = line.rstrip("\n").split("\t")
            chrom = parts[0]
            if not chrom.startswith("chr"):
                chrom = "chr" + chrom.lstrip("ch").lstrip("CH")
            contigs.add(chrom)
            norm.write("\t".join([chrom] + parts[1:]) + "\n")
    norm.flush()
    return norm.name, contigs

def make_windows_bed(df):
    """Write a temp BED of windows; return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".bed")
    for idx, row in df.iterrows():
        tmp.write(f"{row.chrom}\t{int(row.start)}\t{int(row.end)}\t{idx}\n")
    tmp.flush()
    return tmp.name

def intersect_bed(a_bed, b_bed):
    """Run bedtools intersect -wa -wb; return list of output lines."""
    cmd = ["bedtools", "intersect", "-a", a_bed, "-b", b_bed, "-wa", "-wb"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"bedtools error: {proc.stderr}")
    return proc.stdout.splitlines()

def build_mask(df, overlaps):
    """Convert overlaps to a list of 0/1 numpy masks per window."""
    masks = [np.zeros(int(r.end - r.start), dtype=int) for _, r in df.iterrows()]
    for line in overlaps:
        parts = line.split("\t")
        idx      = int(parts[3])
        pk_start = int(parts[5]); pk_end = int(parts[6])
        win_start = int(df.at[idx, "start"])
        rel_s = max(0, pk_start - win_start)
        rel_e = min(len(masks[idx]), pk_end - win_start)
        if rel_s < rel_e:
            masks[idx][rel_s:rel_e] = 1
    return masks

def main():
    # Normalize and load all RBP peak files
    normalized_peaks = {}
    for rbp, path in RBP_PEAKS.items():
        if not os.path.exists(path):
            print(f"ERROR: RBP peak file not found: {path}")
            return
        norm_bed, contigs = normalize_peak_bed(path)
        normalized_peaks[rbp] = norm_bed
        print(f"{rbp} original contigs: {contigs}")

    # Process each CSV
    for csv_file in sorted(glob.glob(CSV_PATTERN)):
        print(f"\nProcessing {csv_file} ...")
        df = pd.read_csv(csv_file)
        # Normalize contigs in your windows
        df["chrom"] = df["chrom"].apply(normalize_chrom_name)
        window_contigs = set(df["chrom"].unique())
        print(f"Window contigs: {window_contigs}")

        win_bed = make_windows_bed(df)

        # For each RBP, intersect & build mask
        for rbp, norm_bed in normalized_peaks.items():
            print(f" - intersecting with {rbp} peaks...")
            overlaps = intersect_bed(win_bed, norm_bed)
            print(f"   → {len(overlaps)} overlaps found for {rbp}")
            mask = build_mask(df, overlaps)
            df[f"{rbp}_bind"] = mask

        os.unlink(win_bed)
        # Overwrite CSV
        df.to_csv(csv_file, index=False)
        print(f"→ Updated {csv_file}")

    # Cleanup normalized peak beds
    for norm_bed in normalized_peaks.values():
        os.unlink(norm_bed)

    print("\nAll done.")

if __name__ == "__main__":
    main()

