import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel

# Define the dataset class
class SpliceSiteDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Filter sequences to exactly 500 nucleotides
        self.data = self.data[self.data['Token'].apply(len) == 500]

        # Precompute targets
        self.targets = self.data.apply(self._create_target_vector, axis=1).values

    def _create_target_vector(self, row):
        """Creates a 500-length binary vector marking splice sites (donor/acceptor)."""
        target = np.zeros(500, dtype=np.float32)
        intron_indices = eval(row['[intron Indices]'])  # Caution: use safer parsing
        
        for intron in intron_indices:
            start, end = intron['start'], intron['end']
            # Donor site (GT at intron start)
            if start + 1 < 500:
                target[start] = 1
                target[start + 1] = 1
            # Acceptor site (AG at intron end)
            if end - 1 >= 0:
                target[end - 1] = 1
                target[end] = 1
        return target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]['Token']
        target = self.targets[idx]

        # Tokenize with DNA BERT (6-mers)
        inputs = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'targets': torch.FloatTensor(target)
        }

# Define the model
class SpliceSiteBERT(nn.Module):
    def __init__(self, bert_model_name="zhihan1996/DNA_bert_6", freeze_bert=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # Freeze BERT weights if needed
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Upsample to 500 positions (DNA BERT uses 6-mers: 500/6 ≈ 83 tokens)
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=self.bert_hidden_size,
                out_channels=128,
                kernel_size=6,
                stride=6,
                padding=0
            ),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.classifier = nn.Conv1d(64, 1, kernel_size=1)

    def forward(self, input_ids, attention_mask):
        # DNA BERT output
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state  # (batch, token_seq_len, 768)

        # Reshape for transposed convolution
        embeddings = embeddings.permute(0, 2, 1)  # (batch, 768, token_seq_len)
        
        # Upsample to ~500 positions (83 tokens * 6 = 498)
        x = self.upsample(embeddings)  # (batch, 64, 498)
        
        # Pad to 500 if needed
        x = torch.nn.functional.pad(x, (0, 2))  # (batch, 64, 500)
        
        # Classify each position
        logits = self.classifier(x).squeeze(1)  # (batch, 500)
        return torch.sigmoid(logits)

# Training function
def train_model(dataloader, model, criterion, optimizer, device="cuda"):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# Evaluation function
def evaluate_model(dataloader, model, criterion, device="cuda"):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Main script
if __name__ == "__main__":
    # Initialize tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6")
    dataset = SpliceSiteDataset("output2.csv", tokenizer)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Initialize model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpliceSiteBERT().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_model(train_loader, model, criterion, optimizer, device)
        val_loss = evaluate_model(test_loader, model, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "dna_bert_splice_model.pth")
    print("Training complete and model saved.")