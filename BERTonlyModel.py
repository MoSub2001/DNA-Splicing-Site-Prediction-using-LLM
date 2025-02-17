import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import ast
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

###############################################################################
# Utility Functions: Parsing & Target Vector Creation
###############################################################################
def parse_exon_indices(value):
    """
    Safely parse exon indices from a string.
    Expected format: a string-encoded list of dictionaries,
    e.g. "[{'start': 0, 'end': 499}, {'start': 100, 'end': 499}, ...]"
    Returns a list of dictionaries.
    """
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
        else:
            return []
    except (ValueError, SyntaxError):
        return []

def create_target_vector(exon_indices_str, sequence_length=500):
    """
    Creates a binary vector of length `sequence_length` where only the
    positions corresponding to splice sites are marked as 1.
    For each dictionary in the exon indices list, we mark the position given by the "start" key.
    (For example, if the indices are [{'start': 0, 'end': ...}, {'start': 100, ...}, ...],
    then positions 0, 100, etc. will be set to 1.)
    """
    indices_list = parse_exon_indices(exon_indices_str)
    target = [0] * sequence_length
    for d in indices_list:
        pos = d.get('start', None)
        if pos is not None and pos < sequence_length:
            target[pos] = 1
    return target

###############################################################################
# Dataset Definition
###############################################################################
class SpliceDataset(Dataset):
    """
    Dataset that:
      - Reads a CSV file containing a 500-character DNA sequence (column "Token")
        and exon indices (column "[exon Indices]").
      - Creates a target vector of length 500 where positions of splice sites (from exon indices)
        are marked as 1.
      - Tokenizes the DNA sequence using a provided BERT tokenizer.
    """
    def __init__(self, csv_file, tokenizer, max_length=500):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.sequences = self.data['Token']
        self.exon_indices = self.data['[exon Indices]']
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences.iloc[idx]  # DNA sequence (string) of length 500
        exon_indices_str = self.exon_indices.iloc[idx]
        # Create target vector: ones only at splice site positions (e.g., 0,100,200,300)
        target_vector = create_target_vector(exon_indices_str, sequence_length=self.max_length)
        
        # Tokenize the sequence.
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length  # Note: DNA-BERT may tokenize into k-mers (e.g., ~83 tokens)
        )
        
        # Remove extra batch dimension.
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target': torch.tensor(target_vector, dtype=torch.float)
        }

###############################################################################
# Model Definition
###############################################################################
class SplicePredictorFromBERT(nn.Module):
    """
    This model uses DNA-BERT to generate embeddings for the input sequence.
    The BERT output (which might be, e.g., 83 token embeddings for k-mer tokenization)
    is flattened into a single vector. A feed-forward network then maps this flattened
    vector to an output vector of length 500 (one prediction per nucleotide position).
    
    Parameters:
      - embedding_dim: Dimension of BERT embeddings (default 768).
      - output_length: Desired output length (500).
      - hidden_dim: Hidden layer dimension.
      - bert_token_count: Expected number of tokens from the BERT tokenizer.
          (For DNA-BERT with 6-mer tokenization, typically ~83 tokens. If you switch to a
           character-level tokenizer, set this to 500.)
    """
    def __init__(self, embedding_dim=768, output_length=500, hidden_dim=256, bert_token_count=83):
        super(SplicePredictorFromBERT, self).__init__()
        self.bert = BertModel.from_pretrained("zhihan1996/DNA_bert_6")
        # Optionally freeze BERT parameters:
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.bert_token_count = bert_token_count
        flattened_dim = self.bert_token_count * embedding_dim
        
        # Fully connected network to map the flattened BERT embeddings to 500 predictions.
        self.fc1 = nn.Linear(flattened_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_length)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        # Obtain BERT embeddings; shape: (batch, seq_len, embedding_dim)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        batch_size, seq_len, emb_dim = bert_out.shape
        
        # Adjust token dimension if needed.
        if seq_len != self.bert_token_count:
            if seq_len > self.bert_token_count:
                bert_out = bert_out[:, :self.bert_token_count, :]
            else:
                pad_size = self.bert_token_count - seq_len
                pad_tensor = torch.zeros(batch_size, pad_size, emb_dim, device=bert_out.device)
                bert_out = torch.cat([bert_out, pad_tensor], dim=1)
        
        # Flatten token embeddings into one vector per sample.
        flat = bert_out.view(batch_size, -1)
        # Pass through fully connected layers.
        x = self.relu(self.fc1(flat))
        x = self.sigmoid(self.fc2(x))
        # Output shape: (batch, 500)
        return x

###############################################################################
# Training Function
###############################################################################
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10, checkpoint_path="model_checkpoint.pth", patience=3):
    model.train()
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        epochs_without_improvement = checkpoint['epochs_without_improvement']
    
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)  # shape: (batch, 500)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)  # shape: (batch, 500)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
        # Checkpointing
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'epochs_without_improvement': epochs_without_improvement
            }, checkpoint_path)
            print("Checkpoint saved")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping")
                break

###############################################################################
# Evaluation Function
###############################################################################
def evaluate_model(model, test_loader, device, threshold=0.7):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)  # shape: (batch, 500)
            
            outputs = model(input_ids, attention_mask)  # shape: (batch, 500)
            predictions = (outputs > threshold).float()  # binary predictions
            correct += (predictions == targets).sum().item()
            total += targets.numel()
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")

###############################################################################
# Main Execution
###############################################################################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("zhihan1996/DNA_bert_6")
    
    # Create the dataset from the CSV file.
    dataset = SpliceDataset("output.csv", tokenizer, max_length=500)
    
    # Split dataset into training (80%) and testing (20%)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # For DNA-BERT with k-mer tokenization, the expected token count is typically ~83.
    # (If you switch to a character-level tokenizer that treats each nucleotide separately,
    #  set bert_token_count=500.)
    bert_token_count = 83
    model = SplicePredictorFromBERT(embedding_dim=768, output_length=500, hidden_dim=256, bert_token_count=bert_token_count)
    model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    checkpoint_path = "model_checkpoint.pth"
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=10, checkpoint_path=checkpoint_path, patience=3)
    evaluate_model(model, test_loader, device, threshold=0.7)
