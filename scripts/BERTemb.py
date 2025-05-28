import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────────────────────────────────────
# Dataset class for loading sequences from CSV
# ─────────────────────────────────────────────────────────────────────────────
class TokenSequenceDataset(Dataset):
    def __init__(self, dataframe):
        self.sequences = dataframe['Token']

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences.iloc[idx]

# ─────────────────────────────────────────────────────────────────────────────
# Generate embeddings and save batch-by-batch to avoid OOM
# ─────────────────────────────────────────────────────────────────────────────
def generate_embeddings_for_chromosome(chrom_num, tokenizer, model, device, input_dir="../chromosomeData", output_dir="bert_embeddings"):
    csv_path = os.path.join(input_dir, f"ch{chrom_num}.csv")
    batch_dir = os.path.join(output_dir, f"batches_ch{chrom_num}")
    os.makedirs(batch_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"⚠️ Missing {csv_path}, skipping.")
        return

    df = pd.read_csv(csv_path)
    print(f"🔍 ch{chrom_num}: {len(df)} sequences loaded")
    dataset = TokenSequenceDataset(df)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model.eval()
    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(loader, desc=f"ch{chrom_num}")):
            enc = tokenizer(
                list(batch),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=504,
                add_special_tokens=False
            ).to(device)

            out = model(**enc)[0].cpu()  # shape: (B, 504, 768)
            torch.save(out, os.path.join(batch_dir, f"batch_{batch_id}.pt"))

    print(f"✅ Saved all batches for ch{chrom_num} to {batch_dir}")

# ─────────────────────────────────────────────────────────────────────────────
# Main runner: iterate over chromosomes
# ─────────────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True).to(device)

    os.makedirs("bert_embeddings", exist_ok=True)

    for chrom in range(1, 23):
        generate_embeddings_for_chromosome(
            chrom_num=chrom,
            tokenizer=tokenizer,
            model=model,
            device=device
        )

    print("🎉 All BERT embeddings saved in batches under: bert_embeddings/")

if __name__ == "__main__":
    print("starting BERTemb.py\n")
    main()
