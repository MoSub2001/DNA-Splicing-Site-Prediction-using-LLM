import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------- Dataset Class -------------------------

class SpliceSiteDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name="zhihan1996/DNABERT-2-117M", seq_len=500):
        self.data = pd.read_csv(csv_file)

        # Ensure sequences are exactly 500 bp
        self.data = self.data[self.data['Token'].apply(len) == seq_len]

        # Transformer tokenizer (DNABERT)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, 
            do_lower_case=False, 
            add_special_tokens=False  # We handle special tokens manually
        )

        self.seq_len = seq_len

        # Select columns containing exon intervals
        self.interval_cols = [
            "exon_positions"  # Replace with your actual exon interval column
        ]

        self.samples = []
        for idx, row in self.data.iterrows():
            seq = row["Token"]
            interval_strs = [row[col] if col in row else "[]" for col in self.interval_cols]
            label_vec = self.create_union_label(interval_strs)

            self.samples.append((seq, label_vec))

    def create_union_label(self, interval_strs):
        """Creates a 500-length label vector with 1s for exon regions, 0s for intron regions."""
        label_vec = torch.zeros(self.seq_len, dtype=torch.float32)
        for interval_str in interval_strs:
            intervals = self.parse_intervals(interval_str)
            for (start, end) in intervals:
                start = max(0, min(self.seq_len - 1, start))
                end = max(0, min(self.seq_len - 1, end))
                label_vec[start:end + 1] = 1.0
        return label_vec

    @staticmethod
    def parse_intervals(interval_str):
        """Parses a string like "[{'start':..., 'end':...}]" into a list of (start, end) tuples."""
        if not isinstance(interval_str, str) or interval_str.strip() == "[]":
            return []
        try:
            interval_str_json = interval_str.replace("'", '"')  # Fix invalid JSON
            intervals = json.loads(interval_str_json)
            return [(d["start"], d["end"]) for d in intervals]
        except:
            return []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence_str, label_vec = self.samples[idx]

        # Tokenize using DNABERT
        tokenized = self.tokenizer(
            sequence_str,
            return_tensors="pt"
        )

        input_ids = tokenized["input_ids"].squeeze(0)  # Shape (seq_len,)
        attention_mask = tokenized["attention_mask"].squeeze(0)  # Shape (seq_len,)

        return input_ids, attention_mask, label_vec

# ------------------------- Transformer Model -------------------------

class SpliceSiteTransformer(nn.Module):
    def __init__(self, transformer_name="zhihan1996/DNABERT-2-117M"):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(transformer_name)
        hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 1)  # Predicts probability per position
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden = outputs.last_hidden_state  # Shape: (batch, seq_len, hidden_size)
        logits = self.classifier(last_hidden).squeeze(-1)  # Shape: (batch, seq_len)
        probs = torch.sigmoid(logits)  # Convert to probability
        return probs

# ------------------------- Training & Evaluation Functions -------------------------

def train(model, data_loader, criterion, optimizer, num_epochs=5):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for input_ids, attention_mask, labels in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

def evaluate(model, data_loader):
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask)
            preds = (outputs > 0.5).float()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Flatten and compute accuracy
    accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# ------------------------- Main Script -------------------------

if __name__ == "__main__":
    csv_file = "output2.csv"  # Replace with actual file
    batch_size = 2 # Reduce if running on CPU
    num_epochs = 5
    learning_rate = 1e-5

    # Load dataset
    dataset = SpliceSiteDataset(csv_file)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = SpliceSiteTransformer().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, train_loader, criterion, optimizer, num_epochs=num_epochs)

    # Evaluate the model
    evaluate(model, test_loader)

    # Save the trained model
    torch.save(model.state_dict(), "splice_site_transformer.pth")
    print("Training complete. Model saved.")
