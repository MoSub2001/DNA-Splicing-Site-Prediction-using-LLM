import csv
import gzip
from Bio import SeqIO
import os

def load_chromosome_17_fasta(fasta_path):
    """
    Loads chromosome 17 from the given FASTA file into a BioPython Seq object.
    Assumes the FASTA has an entry named '17' or 'chr17'.
    """
    # Convert to dict of SeqRecords keyed by the FASTA ID (e.g., "17" or "chr17")
    seq_dict = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
    
    # Figure out the key for chromosome 17
    # Might be "17" or "chr17" depending on your file. Adjust if needed.
    if "17" in seq_dict:
        return seq_dict["17"].seq
    elif "chr17" in seq_dict:
        return seq_dict["chr17"].seq
    else:
        raise ValueError("Chromosome 17 not found in the given FASTA.")


def extract_gene_info_from_gtf(gtf_path, gene_name_of_interest):
    """
    Reads a GTF (gzipped) to find the chromosome, start, end, and strand
    for a gene with the specified gene_name.
    Returns a dict with keys: chromosome, start, end, strand or None if not found.
    """
    gene_info = None
    with gzip.open(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            feature_type = fields[2]
            
            if feature_type == "gene":
                # Example attributes: gene_name "ELP5"; gene_id "ENSG..."
                attr_part = fields[8]
                # Parse attributes
                attributes = {}
                for attr in attr_part.split(";"):
                    attr = attr.strip()
                    if not attr:
                        continue
                    key_val = attr.split(" ")
                    if len(key_val) >= 2:
                        k = key_val[0]
                        v = key_val[1].strip('"')
                        attributes[k] = v
                
                if attributes.get("gene_name") == gene_name_of_interest:
                    gene_info = {
                        "chromosome": fields[0],  # might be '17' or 'chr17'
                        "start": int(fields[3]),
                        "end": int(fields[4]),
                        "strand": fields[6]
                    }
                    break
    return gene_info


def extract_features_from_gtf(gtf_path, gene_name, gene_start, gene_end, chromosome):
    """
    Returns a dict of feature lists:
      {
        "exon": [ {start, end}, ... ],
        "five_prime_utr": [ {start, end}, ... ],
        "three_prime_utr": [ {start, end}, ... ],
        ...
      }
    We only gather items that match the given gene_name, chromosome, and lie
    within [gene_start, gene_end].
    """
    features = {
        "exon": [],
        "five_prime_utr": [],
        "three_prime_utr": [],
    }
    
    with gzip.open(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            chrom = fields[0]
            ftype = fields[2]  # e.g. 'exon', 'five_prime_UTR', 'three_prime_UTR'
            start_ = int(fields[3])
            end_ = int(fields[4])
            attr_part = fields[8]
            
            # Skip if not same chromosome
            if chrom != chromosome:
                continue
            
            # We only care about exons and UTRs
            if ftype not in ["exon", "five_prime_utr", "three_prime_utr"]:
                continue
            
            # Parse attributes to check gene_name
            attributes = {}
            for attr in attr_part.split(";"):
                attr = attr.strip()
                if not attr:
                    continue
                key_val = attr.split(" ", 1)
                if len(key_val) == 2:
                    k = key_val[0]
                    v = key_val[1].strip('"')
                    attributes[k] = v
            
            if attributes.get("gene_name") != gene_name:
                continue
            
            # Check overlap with gene region
            if end_ < gene_start or start_ > gene_end:
                continue
            
            # Clip to gene boundaries if desired:
            start_ = max(start_, gene_start)
            end_ = min(end_, gene_end)
            
            features[ftype].append({"start": start_, "end": end_})
    
    # Sort them by start
    for ftype in features:
        features[ftype].sort(key=lambda x: x["start"])
    
    return features


def deduplicate_feature_intervals(feature_list):
    """
    Given a list of dicts like [{'start': int, 'end': int}, ...],
    remove duplicates based on (start, end).
    """
    seen = set()
    deduped = []
    for f in feature_list:
        key = (f["start"], f["end"])
        if key not in seen:
            seen.add(key)
            deduped.append(f)
    return deduped


# i love u u love me 



def calculate_introns(exons):
    """
    Given a list of exons (each a dict with start, end), sorted by start,
    return a list of introns (start, end).
    """
    introns = []
    if not exons:
        return introns
    
    # Ensure sorted by start
    exons = sorted(exons, key=lambda x: x["start"])
    
    for i in range(len(exons) - 1):
        intron_start = exons[i]["end"] + 1
        intron_end = exons[i+1]["start"] - 1
        if intron_start <= intron_end:
            introns.append({"start": intron_start, "end": intron_end})
    return introns


def extract_enhancers_promoters_from_gff3(gff3_path, chromosome, gene_start, gene_end):
    """
    Parse a GFF3 (gzipped) that presumably has lines with type='enhancer' or type='promoter'
    in the 3rd column. Return a dict:
        {
           "enhancer": [ {start, end}, ...],
           "promoter": [ {start, end}, ...]
        }
    filtered to the specified chromosome region.
    """
    features = {
        "enhancer": [],
        "promoter": []
    }
    
    with gzip.open(gff3_path, "rt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.strip().split("\t")
            chrom = fields[0]
            ftype = fields[2]  # e.g. 'enhancer', 'promoter', 'CTCF_binding_site', etc.
            start_ = int(fields[3])
            end_ = int(fields[4])
            
            # Filter by chromosome
            if chrom != chromosome:
                continue
            
            # Check overlap with gene region
            if end_ < gene_start or start_ > gene_end:
                continue
            
            if ftype in ["enhancer", "Enhancer"]:
                # Clip if needed
                s = max(start_, gene_start)
                e = min(end_, gene_end)
                features["enhancer"].append({"start": s, "end": e})
            elif ftype in ["promoter", "Promoter"]:
                s = max(start_, gene_start)
                e = min(end_, gene_end)
                features["promoter"].append({"start": s, "end": e})
    
    # Sort
    for k in features:
        features[k].sort(key=lambda x: x["start"])
    
    return features


def extract_polyA_signals_from_atlas_bed(
    bed_path, chromosome, gene_start, gene_end, strand=None
):
    """
    Parses the 'atlas.clusters.3.0.GRCh38.GENCODE_42.bed' file (or similar),
    extracting lines overlapping [gene_start, gene_end] on the specified chromosome.
    If 'strand' is provided, we filter by that strand as well.

    Returns a list of dicts: [ { 'start': int, 'end': int }, ... ].
    """
    sites = []

    # Detect if gzipped
    is_gz = bed_path.endswith(".gz")
    open_fn = gzip.open if is_gz else open
    mode = "rt" if is_gz else "r"

    with open_fn(bed_path, mode) as f:
        for line in f:
            # Skip headers or empty lines
            if line.startswith("#") or not line.strip():
                continue
            fields = line.strip().split("\t")
            if len(fields) < 6:
                continue  # not a valid 6-column BED line

            bed_chrom = fields[0]  # e.g. 'chr17'
            bed_start = int(fields[1]) + 1  # Convert 0-based -> 1-based
            bed_end = int(fields[2])       # Typically non-inclusive end
            bed_strand = fields[5]         # +, -, or .

            # Filter by chromosome
            # The chromosome in your gene_info might be '17' or 'chr17';
            # adapt if needed:
            if bed_chrom == chromosome or bed_chrom == "chr" + chromosome:
                # Check overlap with [gene_start, gene_end]
                if bed_end < gene_start or bed_start > gene_end:
                    continue  # no overlap

                # If user wants to match strand
                if strand and bed_strand != strand and bed_strand != ".":
                    continue

                # Clip to gene boundaries if you want
                s = max(bed_start, gene_start)
                e = min(bed_end, gene_end)
                if s <= e:
                    sites.append({"start": s, "end": e})

    # Sort by start
    sites.sort(key=lambda x: x["start"])
    return sites

def tokenize_and_write_csv(sequence, features, gene_start, output_csv, token_length=500, max_lines=10000):
    """
    Appends data for a new gene to the existing CSV. Stops if max_lines is reached.
    """
    # Convert features to relative coords
    rel_features = {}
    for ftype in features:
        rel_features[ftype] = []
        for region in features[ftype]:
            start_ = region["start"] - gene_start
            end_ = region["end"] - gene_start
            rel_features[ftype].append({"start": start_, "end": end_})

    # List of feature types in the CSV
    header_feature_list = [
        "exon", "intron", "five_prime_utr", "three_prime_utr",
        "enhancer", "promoter", "polyA_signal"
    ]

    file_exists = os.path.exists(output_csv)
    current_line_count = 0

    if file_exists:
        # Get the current line count of the CSV
        with open(output_csv, mode="r") as f:
            current_line_count = sum(1 for _ in f)

    with open(output_csv, mode="a", newline="") as out:
        writer = csv.writer(out)
        
        # Write header if the file is new or empty
        if not file_exists or os.stat(output_csv).st_size == 0:
            header = ["Token"]
            for ftype in header_feature_list:
                header.append(f"[{ftype} Indices]")
            for ftype in header_feature_list:
                header.append(f"[{ftype} # ]")

            writer.writerow(header)
            current_line_count += 1
        
        # Split sequence into tokens
        tokens = [sequence[i : i + token_length] for i in range(0, len(sequence), token_length)]
        
        for token_idx, token in enumerate(tokens):
            if current_line_count >= max_lines:
                print(f"Reached max line limit ({max_lines}). Stopping...")
                return current_line_count

            token_start = token_idx * token_length
            token_end = token_start + len(token)

            # Collect overlaps (with dedup)
            overlap_dict = {}
            for ftype in header_feature_list:
                overlap_dict[ftype] = []
                seen_coords = set()
                
                for region in rel_features.get(ftype, []):
                    if region["end"] < token_start or region["start"] > token_end - 1:
                        continue
                    clipped_start = max(region["start"], token_start) - token_start
                    clipped_end = min(region["end"], token_end - 1) - token_start
                    coord_key = (clipped_start, clipped_end)

                    # Only add if unique
                    if coord_key not in seen_coords:
                        seen_coords.add(coord_key)
                        overlap_dict[ftype].append({"start": clipped_start, "end": clipped_end})

            # Build CSV row
            row = [token]
            for ftype in header_feature_list:
                row.append(overlap_dict[ftype])
            for ftype in header_feature_list:
                row.append(len(overlap_dict[ftype]))
            
            writer.writerow(row)
            current_line_count += 1

    return current_line_count


def extract_genes_on_chromosome(gtf_file, chromosome="17"):
    """
    Extracts a list of all unique gene names on a specified chromosome
    from a GTF file.
    
    Args:
        gtf_file (str): Path to the GTF file.
        chromosome (str): Chromosome of interest (e.g., "17" or "chr17").
        
    Returns:
        set: A set of gene names located on the specified chromosome.
    """
    genes = set()

    with gzip.open(gtf_file, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue  # Skip header lines
            fields = line.strip().split("\t")
            if len(fields) < 9:
                continue  # Skip malformed lines

            # Check if the line is for a gene on the desired chromosome
            chrom = fields[0]  # Chromosome (e.g., "17" or "chr17")
            feature_type = fields[2]  # Feature type (e.g., "gene")
            if chrom == chromosome and feature_type == "gene":
                # Parse the attributes field to extract the gene name
                attributes = fields[8]
                for attr in attributes.split(";"):
                    attr = attr.strip()
                    if attr.startswith("gene_name"):
                        gene_name = attr.split(" ")[1].strip('"')
                        genes.add(gene_name)
                        break  # Stop after finding gene_name

    return genes

# Example usage




def main():
    # -------------------------------------------------------------------
    # 1. FILE PATHS & PARAMETERS
    # -------------------------------------------------------------------
    # Example file paths (adjust as needed!)
    chromosome_17_fasta = "Homo_sapiens.GRCh38.dna.chromosome.17.fa"
    gtf_file = "data/Homo_sapiens.GRCh38.113.gtf.gz"   # GTF annotation file
    gff3_file = "Homo_sapiens.GRCh38.regulatory_features.v113 (1).gff3.gz"  # GFF3 for enhancers/promoters
    output_csv = "output2.csv"
    bed_polyA_path = "atlas.clusters.3.0.GRCh38.GENCODE_42.bed"
    max_lines = 10000

    chromosome = "17"  # Adjust if your GTF uses "chr17"
    chrom17_seq = load_chromosome_17_fasta(chromosome_17_fasta)
    genes_on_chr17 = extract_genes_on_chromosome(gtf_file, chromosome)
    print(f"Total genes on chromosome 17: {len(genes_on_chr17)}")

    current_line_count = 0
    for gene_name in genes_on_chr17:
        print(f"Extracting gene info for gene_name='{gene_name}'...")
        gene_info = extract_gene_info_from_gtf(gtf_file, gene_name)
        if not gene_info:
            raise ValueError(f"Could not find gene_name '{gene_name}' in {gtf_file}")
        chromosome = gene_info["chromosome"]  # e.g. '17'
        gene_start, gene_end = gene_info["start"], gene_info["end"]
        strand = gene_info["strand"]

        print("Extracting exons, 5'UTRs, 3'UTRs from GTF...")
        basic_features = extract_features_from_gtf(gtf_file, gene_name, gene_start, gene_end, chromosome)
        print("Calculating introns from exons...")
        introns = calculate_introns(basic_features["exon"])
        print("Extracting enhancers & promoters from GFF3...")
        reg_features = extract_enhancers_promoters_from_gff3(gff3_file, chromosome, gene_start, gene_end)

        all_features = {
            "exon": basic_features["exon"],
            "five_prime_utr": basic_features["five_prime_utr"],
            "three_prime_utr": basic_features["three_prime_utr"],
            "intron": introns,
            "enhancer": reg_features["enhancer"],
            "promoter": reg_features["promoter"],
            "polyA_signal": extract_polyA_signals_from_atlas_bed(
                bed_polyA_path, chromosome, gene_start, gene_end, strand
            ),
        }

        # Deduplicate features
        for ftype in all_features:
            all_features[ftype] = deduplicate_feature_intervals(all_features[ftype])

        gene_seq = chrom17_seq[gene_start - 1 : gene_end]
        if strand == "-":
            gene_seq = gene_seq.reverse_complement()
        gene_seq = str(gene_seq)

        print(f"Tokenizing and writing to {output_csv}...")
        current_line_count = tokenize_and_write_csv(gene_seq, all_features, gene_start, output_csv, max_lines=max_lines)
        if current_line_count >= max_lines:
            print(f"Reached {max_lines} lines. Stopping further processing.")
            break

    print("Done!")

if __name__ == "__main__":
    main()
