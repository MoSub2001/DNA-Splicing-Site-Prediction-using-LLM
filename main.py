import csv
from Bio import SeqIO
import gzip

# File paths
fasta_file = "Homo_sapiens.GRCh38.dna.chromosome.17.fa"
gtf_file = "Homo_sapiens.GRCh38.113.gtf.gz"
output_csv = "TP53_exon_intron_indices.csv"

# Load the FASTA file
def load_genome(fasta_path):
    genome = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
    return genome

# Extract a sequence given chromosome and positions
def extract_sequence(genome, chromosome, start, end, strand):
    sequence = genome[chromosome].seq[start-1:end]  # Convert to 0-based indexing
    if strand == "-":
        sequence = sequence.reverse_complement()
    return str(sequence)

# Extract exon positions from GTF
def extract_exons(gtf_path, gene_name, gene_start, gene_end):
    exon_positions = []

    with gzip.open(gtf_path, "rt") as file:
        for line in file:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if fields[2] == "exon":
                attributes = {key_value.split(" ")[0]: key_value.split(" ")[1].strip('"') for key_value in [attr.strip() for attr in fields[8].split(";") if attr]}
                if attributes.get("gene_name") == gene_name:
                    exon_start = int(fields[3])
                    exon_end = int(fields[4])
                    if exon_start >= gene_start and exon_end <= gene_end:
                        exon_positions.append({"start": exon_start, "end": exon_end})

    exon_positions.sort(key=lambda x: x["start"])
    return exon_positions

# Extract gene information from GTF
def extract_gene_annotations(gtf_path, gene_name_of_interest):
    gene_info = None
    with gzip.open(gtf_path, "rt") as file:
        for line in file:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if fields[2] == "gene":
                attributes = {key_value.split(" ")[0]: key_value.split(" ")[1].strip('"') for key_value in [attr.strip() for attr in fields[8].split(";") if attr]}
                if attributes.get("gene_name") == gene_name_of_interest:
                    gene_info = {
                        "chromosome": fields[0],
                        "start": int(fields[3]),
                        "end": int(fields[4]),
                        "strand": fields[6]
                    }
                    break
    return gene_info

# Calculate introns based on exons
def calculate_introns(exons):
    intron_positions = []
    for i in range(len(exons) - 1):
        intron_start = exons[i]["end"] + 1
        intron_end = exons[i + 1]["start"] - 1
        intron_positions.append({"start": intron_start, "end": intron_end})
    return intron_positions

def tokenize_and_append(sequence, exons, introns, gene_start, output_csv, token_length=500):
    """
    Tokenizes the gene sequence and appends exon/intron indices relative to the gene start to the existing CSV file.

    Args:
        sequence (str): The gene sequence.
        exons (list): List of exon start and end positions relative to the genome.
        introns (list): List of intron start and end positions relative to the genome.
        gene_start (int): Start position of the gene in the genome.
        output_csv (str): Path to the output CSV file.
        token_length (int): Length of each token.
    """
    # Adjust exons and introns to be relative to the gene start
    relative_exons = [{"start": exon["start"] - gene_start, "end": exon["end"] - gene_start} for exon in exons]
    relative_introns = [{"start": intron["start"] - gene_start, "end": intron["end"] - gene_start} for intron in introns]
    
    # Ensure valid ranges
    relative_exons = [e for e in relative_exons if e["start"] <= e["end"]]
    relative_introns = [i for i in relative_introns if i["start"] <= i["end"]]

    with open(output_csv, mode="a", newline="") as file:  # Open in append mode
        writer = csv.writer(file)

        # Check if file is empty and write header only if needed
        file.seek(0, 2)  # Move to the end of the file
        if file.tell() == 0:  # File is empty
            writer.writerow(["Token", "[Intron Indices]", "Intron #", "[Exon Indices]", "Exon #"])

        # Tokenize the sequence
        tokens = [sequence[i:i + token_length] for i in range(0, len(sequence), token_length)]
        for token_index, token in enumerate(tokens):
            token_start = token_index * token_length
            token_end = token_start + len(token)

            # Find intron indices relative to the token
            intron_indices = [
                {"start": max(intron["start"] - token_start, 0),
                 "end": min(intron["end"] - token_start, len(token))}
                for intron in relative_introns
                if token_start <= intron["end"] and token_end >= intron["start"]
            ]
            unique_intron_indices = list({(i["start"], i["end"]): i for i in intron_indices}.values())  # Deduplicate

            # Find exon indices relative to the token
            exon_indices = [
                {"start": max(exon["start"] - token_start, 0),
                 "end": min(exon["end"] - token_start, len(token))}
                for exon in relative_exons
                if token_start <= exon["end"] and token_end >= exon["start"]
            ]
            unique_exon_indices = list({(e["start"], e["end"]): e for e in exon_indices}.values())  # Deduplicate

            writer.writerow([
                token,
                unique_intron_indices,
                len(unique_intron_indices),
                unique_exon_indices,
                len(unique_exon_indices)
            ])
            # print(f"Token {token_index}: Start {token_start}, End {token_end}, "
            #       f"Unique Introns: {unique_intron_indices}, Unique Exons: {unique_exon_indices}")

# Main processing for ERBB2
genome = load_genome(fasta_file)
gene_name = "CHK1"
gene_info = extract_gene_annotations(gtf_file, gene_name)

if gene_info:
    chromosome = gene_info["chromosome"]
    start, end, strand = gene_info["start"], gene_info["end"], gene_info["strand"]
    sequence = extract_sequence(genome, chromosome, start, end, strand)
    exons = extract_exons(gtf_file, gene_name, start, end)
    introns = calculate_introns(exons)
    
    # # Debugging outputs
    # print(f"Gene Sequence Length: {len(sequence)}")
    # print(f"Exons: {exons}")
    # print(f"Introns: {introns}")

    tokenize_and_append(sequence, exons, introns, start, output_csv)

print(f"Data for {gene_name} saved to {output_csv}.")
