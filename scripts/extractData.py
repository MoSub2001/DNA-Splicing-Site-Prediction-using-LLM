#!/usr/bin/env python3
"""
Script to compute 6-mer probabilities around exon splice sites (both start and end).
Window: 200 bases before + 100 bases after each splice site.
"""

import gzip
from collections import Counter
import pandas as pd
from Bio import SeqIO

# === CONFIGURATION ===
FASTA_PATH = "data/Homo_sapiens.GRCh38.dna_sm.chromosome.2.fa.gz"
GTF_PATH = "data/Homo_sapiens.GRCh38.113.gtf.gz"
CHROM_NAME = "2"  # Adjust if GTF uses "chr2"
KMER_SIZE = 6
UPSTREAM = 200
DOWNSTREAM = 100
OUTPUT_CSV = "kmer_probabilities_2.csv"

# === LOAD CHROMOSOME SEQUENCE ===
print(f"Loading chromosome {CHROM_NAME} sequence...")
with gzip.open(FASTA_PATH, "rt") as f:
    record = next(SeqIO.parse(f, "fasta"))
    chr_seq = record.seq.upper()  # ✅ Force chromosome sequence to uppercase
chr_length = len(chr_seq)

# === HELPER FUNCTION ===
def extract_kmers(window_seq, splice_position, k=6):
    """Yield (k-mer, relative position to splice site) from a window."""
    window_seq = window_seq.upper()  # ✅ Force window sequence to uppercase
    for i in range(len(window_seq) - k + 1):
        kmer = window_seq[i:i+k]
        relative_pos = i - UPSTREAM
        yield kmer, relative_pos

# === GLOBAL COUNTERS ===
kmer_position_counts = Counter()
position_totals = Counter()

# === PARSE GTF AND PROCESS SPLICE SITES ===
print(f"Scanning GTF {GTF_PATH}...")
with gzip.open(GTF_PATH, "rt") as gtf_file:
    for line in gtf_file:
        if line.startswith("#"):
            continue
        cols = line.strip().split("\t")
        if len(cols) < 9:
            continue
        seqname, source, feature, start, end, score, strand, frame, attributes = cols
        if seqname not in {CHROM_NAME, f"chr{CHROM_NAME}"}:
            continue
        if feature != "exon":
            continue

        start, end = int(start), int(end)

        # === Process exon start (5' splice site) ===
        splice_start = start
        window_start = splice_start - UPSTREAM
        window_end = splice_start + DOWNSTREAM - 1

        if window_start >= 1 and window_end <= chr_length:
            window_seq = chr_seq[window_start-1:window_end]
            for kmer, rel_pos in extract_kmers(window_seq, splice_start):
                kmer_position_counts[(kmer, rel_pos)] += 1
                position_totals[rel_pos] += 1

        # === Process exon end (3' splice site) ===
        splice_end = end
        window_start = splice_end - UPSTREAM
        window_end = splice_end + DOWNSTREAM - 1

        if window_start >= 1 and window_end <= chr_length:
            window_seq = chr_seq[window_start-1:window_end]
            for kmer, rel_pos in extract_kmers(window_seq, splice_end):
                kmer_position_counts[(kmer, rel_pos)] += 1
                position_totals[rel_pos] += 1

# === COMPUTE PROBABILITIES ===
print("Computing k-mer probabilities...")
records = []
for (kmer, pos), count in kmer_position_counts.items():
    total = position_totals[pos]
    probability = count / total
    records.append((kmer, pos, probability))

# === SAVE TO CSV ===
df_out = pd.DataFrame(records, columns=["kmer", "distance", "probability"])
df_out = df_out.sort_values(by=["distance", "kmer"]).reset_index(drop=True)
df_out.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Done! K-mer probabilities saved to {OUTPUT_CSV}")
