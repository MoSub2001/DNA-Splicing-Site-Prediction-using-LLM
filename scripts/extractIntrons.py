#!/usr/bin/env python3
"""
Compute 6-mer probabilities around intron boundaries (start and end) on chromosome 2.
Each window is 200 nt upstream and 100 nt downstream of the splice site.
"""

import gzip
from collections import defaultdict, Counter
import pandas as pd
from Bio import SeqIO

# === CONFIGURATION ===
FASTA_PATH = "data/Homo_sapiens.GRCh38.dna_sm.chromosome.2.fa.gz"
GTF_PATH = "data/Homo_sapiens.GRCh38.113.gtf.gz"
CHROM_NAME = "2"
KMER_SIZE = 6
UPSTREAM = 200
DOWNSTREAM = 100
OUTPUT_CSV = "kmer_probabilities_introns_chr2.csv"

# === LOAD CHROMOSOME SEQUENCE ===
print(f"Loading chromosome {CHROM_NAME}...")
with gzip.open(FASTA_PATH, "rt") as f:
    record = next(SeqIO.parse(f, "fasta"))
    chr_seq = record.seq.upper()
chr_length = len(chr_seq)

# === PARSE GTF: Collect exons per transcript on chr2 ===
print("Parsing GTF to extract introns...")
transcripts = defaultdict(list)

with gzip.open(GTF_PATH, "rt") as gtf:
    for line in gtf:
        if line.startswith("#"):
            continue
        cols = line.strip().split("\t")
        if len(cols) < 9:
            continue
        seqname, _, feature, start, end, _, strand, _, attributes = cols
        if seqname not in {CHROM_NAME, f"chr{CHROM_NAME}"} or feature != "exon":
            continue
        attr_dict = dict(item.strip().split(" ", 1) for item in attributes.strip(";").split("; ") if " " in item)
        transcript_id = attr_dict.get("transcript_id", "").strip('"')
        transcripts[transcript_id].append((int(start), int(end), strand))

# === HELPER: Slide kmers and yield relative positions ===
def extract_kmers(window_seq, k=6):
    for i in range(len(window_seq) - k + 1):
        kmer = window_seq[i:i+k]
        rel_pos = i - UPSTREAM
        yield kmer, rel_pos

# === COUNT KMERS AROUND INTRON SPLICE SITES ===
kmer_counts = Counter()
position_totals = Counter()

for tx_id, exons in transcripts.items():
    if not exons or len(exons) < 2:
        continue
    # Sort exons by genomic position
    exons_sorted = sorted(exons, key=lambda x: x[0])
    for i in range(len(exons_sorted) - 1):
        _, exon1_end, strand = exons_sorted[i]
        exon2_start, _, _ = exons_sorted[i+1]

        # Define intron
        intron_start = exon1_end + 1
        intron_end = exon2_start - 1

        if intron_start >= intron_end:
            continue

        # === Intron Start (after exon1) ===
        ws = intron_start - UPSTREAM
        we = intron_start + DOWNSTREAM - 1
        if 1 <= ws <= we <= chr_length:
            window_seq = chr_seq[ws-1:we]
            for kmer, pos in extract_kmers(window_seq, KMER_SIZE):
                kmer_counts[(kmer, pos)] += 1
                position_totals[pos] += 1

        # === Intron End (before exon2) ===
        ws = intron_end - UPSTREAM
        we = intron_end + DOWNSTREAM - 1
        if 1 <= ws <= we <= chr_length:
            window_seq = chr_seq[ws-1:we]
            for kmer, pos in extract_kmers(window_seq, KMER_SIZE):
                kmer_counts[(kmer, pos)] += 1
                position_totals[pos] += 1

# === COMPUTE PROBABILITIES ===
print("Computing probabilities...")
records = []
for (kmer, pos), count in kmer_counts.items():
    total = position_totals[pos]
    prob = count / total if total > 0 else 0.0
    records.append((kmer, pos, prob))

# === SAVE TO CSV ===
df = pd.DataFrame(records, columns=["kmer", "distance", "probability"])
df = df.sort_values(by=["distance", "kmer"]).reset_index(drop=True)
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Done! Output saved to {OUTPUT_CSV}")
