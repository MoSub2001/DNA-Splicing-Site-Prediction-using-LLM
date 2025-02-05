###############################################################################
# IMPORTS
###############################################################################
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import ast
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

###############################################################################
# 1. LOAD AND PREPROCESS DATA
###############################################################################
def parse_feature(value):
    """
    Safely parse string-encoded lists/dictionaries into numeric values.

    Parameters:
    -----------
    value : str or numeric
        This can be a string that represents a list or dictionary, or
        it can already be a numeric type.

    Returns:
    --------
    Numeric value (float or int). If the original is a list, it returns
    the length of that list. If it is already numeric, it returns it directly.
    If parsing fails, it returns 0.
    """
    # Check if the input is a string (e.g., "[1, 2, 3]" or "5.0")
    if isinstance(value, str):
        try:
            # Try to parse the string as a Python literal (e.g., list, dict, float)
            parsed = ast.literal_eval(value)
            # If it's a list, return the length. If it's not a list, assume it's a numeric value and cast to float
            return len(parsed) if isinstance(parsed, list) else float(parsed)
        except (ValueError, SyntaxError):
            # If parsing fails, return 0
            return 0
    # If the value was already numeric (not a string), just return it
    return value

class SpliceDataset(Dataset):
    """
    Custom PyTorch Dataset for loading splice data from a CSV file.

    It reads:
      - DNA sequences (under the column name 'Token')
      - A binary label for splicing (under the column name 'donor')
      - Other extra features stored in columns in the CSV

    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing the data.
    tokenizer : transformers.BertTokenizer
        A BERT tokenizer that can tokenize DNA sequences.
    max_length : int, default=500
        The maximum number of tokens for the BERT model to consider.
    """

    def __init__(self, csv_file, tokenizer, max_length=500):
        # Load the CSV into a pandas DataFrame
        self.data = pd.read_csv(csv_file)
        # Save tokenizer and max_length for later use
        self.tokenizer = tokenizer
        self.max_length = max_length

        # The 'Token' column contains DNA sequences that we will tokenize
        self.sequences = self.data['Token']

        # The 'donor' column contains the splice label (binary: 0 or 1)
        self.splice_labels = self.data['donor']

        # Extract all additional feature columns dynamically:
        #   - We take columns from 1 to -3 (excluding the last 3 columns, 
        #     which are 'Token', 'donor', and possibly others if present).
        #   - We apply parse_feature to safely convert strings to numeric values.
        #   - fillna(0) replaces any missing values with 0.
        self.extra_features = (
            self.data.iloc[:, 1:-3]
            .applymap(parse_feature)
            .fillna(0)
            .values
        )
        # extra_feature_dim will help us know how many extra features we have
        self.extra_feature_dim = self.extra_features.shape[1]

    def __len__(self):
        """
        Returns the total number of examples in the dataset.
        """
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Fetch a single sample from the dataset at the given index.

        Returns a dictionary containing:
          - input_ids: token IDs for the DNA sequence
          - attention_mask: indicates which tokens are real vs. padded
          - extra_features: the additional numeric features
          - labels: a vector of length max_length, each entry is the same 
                    binary label (since the model outputs 500 probabilities)
        """
        # Grab the DNA sequence
        sequence = self.sequences[idx]

        # Tokenize the sequence (return_tensors="pt" gives PyTorch tensors)
        inputs = self.tokenizer(
            sequence, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length
        )

        # Convert the splice label to a float tensor (0.0 or 1.0)
        label = torch.tensor(float(self.splice_labels[idx]), dtype=torch.float)
        # Create a label vector of size 'max_length', all containing the same label
        label_vector = torch.full((self.max_length,), label)

        # Extra features for this sample, converted to float tensor
        extra_features = torch.tensor(self.extra_features[idx], dtype=torch.float)

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'extra_features': extra_features,
            'labels': label_vector
        }

###############################################################################
# 2. DEFINE THE MODEL
###############################################################################
class SplicePredictor(nn.Module):
    """
    A neural network for splice prediction that uses a pretrained BERT
    to get embeddings, then appends extra features, and finally passes 
    this combined input through some linear layers.

    Parameters:
    -----------
    embedding_dim : int, default=768
        The size of BERT's [CLS] token embedding. For DNA-BERT, it's typically 768.
    extra_feature_dim : int, default=15
        Number of extra features. This is determined dynamically at runtime.
    hidden_dim : int, default=256
        Dimension of the hidden layer before the final output.
    """

    def __init__(self, embedding_dim=768, extra_feature_dim=15, hidden_dim=256):
        super(SplicePredictor, self).__init__()

        # Load the pretrained DNA-BERT model
        # "zhihan1996/DNA_bert_6" is a DNA-BERT checkpoint
        self.bert = BertModel.from_pretrained("zhihan1996/DNA_bert_6")

        # Freeze all BERT parameters so they do not update during training
        for param in self.bert.parameters():
            param.requires_grad = False

        # The input dimension is the sum of BERT embedding dimension (768) 
        # plus however many extra features we have
        self.input_dim = embedding_dim + extra_feature_dim

        # A fully connected layer to go from input_dim to a hidden dimension
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)

        # Final fully connected layer that outputs 500 values (because we want 
        # a probability for each position along the sequence length)
        self.fc2 = nn.Linear(hidden_dim, 500)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Sigmoid will give us outputs between 0 and 1

    def forward(self, input_ids, attention_mask, extra_features):
        """
        Forward pass of the model.

        Parameters:
        -----------
        input_ids : torch.Tensor
            Token IDs for the input sequences (batch_size x max_length).
        attention_mask : torch.Tensor
            Attention masks (batch_size x max_length).
        extra_features : torch.Tensor
            Additional features (batch_size x extra_feature_dim).

        Returns:
        --------
        A tensor of shape (batch_size x 500) containing probabilities for
        each position in the sequence.
        """
        # Use BERT to get embeddings, but disable gradient computation 
        # to keep BERT frozen
        with torch.no_grad():
            # BERT returns the last hidden state of shape (batch_size x sequence_length x hidden_dim)
            # We only take the [CLS] token at index 0 along the sequence dimension
            bert_output = self.bert(
                input_ids,
                attention_mask=attention_mask
            ).last_hidden_state[:, 0, :]

        # Ensure extra_features is the right shape (batch_size x extra_feature_dim)
        extra_features = extra_features.view(extra_features.size(0), -1)

        # Concatenate the BERT [CLS] embedding with the extra features
        combined_input = torch.cat((bert_output, extra_features), dim=1)

        # Debugging: check the shape
        print(f"Final Combined Input Shape: {combined_input.shape}")

        # Pass through the first fully-connected layer + ReLU
        x = self.relu(self.fc1(combined_input))
        # Pass through the second layer + Sigmoid to get probabilities
        x = self.sigmoid(self.fc2(x))

        return x

###############################################################################
# 3. TRAINING AND EVALUATION
###############################################################################
def train_model(
    model, 
    train_loader, 
    criterion, 
    optimizer, 
    device, 
    num_epochs=10, 
    checkpoint_path="model_checkpoint.pth", 
    patience=3
):
    """
    Train the model with early stopping and checkpoint saving.

    Parameters:
    -----------
    model : nn.Module
        The PyTorch model to train.
    train_loader : DataLoader
        Dataloader object for the training dataset.
    criterion : nn.Module
        The loss function, such as nn.BCELoss.
    optimizer : torch.optim.Optimizer
        The optimizer, e.g. Adam.
    device : torch.device
        A device to train on, either 'cpu' or 'cuda'.
    num_epochs : int, default=10
        How many epochs to train for.
    checkpoint_path : str, default="model_checkpoint.pth"
        Where to save the best model checkpoint.
    patience : int, default=3
        How many epochs without improvement before early stopping.
    """
    # Switch model to training mode
    model.train()

    # Track the best loss achieved so far
    best_loss = float('inf')
    # Track epochs without improvement
    epochs_without_improvement = 0
    # Check if a checkpoint already exists
    if os.path.exists(checkpoint_path):
        print(f"Loading model checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        # Load model and optimizer states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Resume from the next epoch
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        epochs_without_improvement = checkpoint['epochs_without_improvement']
        print(f"Resuming training from epoch {start_epoch}, best loss so far: {best_loss:.4f}")
    else:
        start_epoch = 0

    # Main training loop
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0

        # We iterate through the training data in batches
        for batch in tqdm(train_loader):
            # Move each part of the batch to the appropriate device (CPU or GPU)
            input_ids = batch['input_ids'].to(device) # BERT input IDs
            attention_mask = batch['attention_mask'].to(device) # BERT attention mask
            extra_features = batch['extra_features'].to(device) # Extra features
            labels = batch['labels'].to(device)    # Ground-truth values (in this case, 0 or 1 for donor splice). In this code, each batch returns a label vector of length 500

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass through the model
            outputs = model(input_ids, attention_mask, extra_features)
            print("Model outputs:", outputs)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backprop: compute gradients
            loss.backward()

            # Update weights
            optimizer.step()

            # Accumulate the batch loss to compute an average
            total_loss += loss.item()

        # Average loss for this epoch
        avg_loss = total_loss / len(train_loader)

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Early stopping logic: if this is the best loss so far, save the model 
        # and reset the patience counter
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'epochs_without_improvement': epochs_without_improvement
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path} (New Best Loss: {best_loss:.4f})")
        else:
            # If no improvement, increment the counter
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs")

            # If we've gone 'patience' epochs with no improvement, stop
            if epochs_without_improvement >= patience:
                print(f"Stopping early after {patience} epochs without improvement.")
                break

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on a test dataset.

    Parameters:
    -----------
    model : nn.Module
        A trained PyTorch model.
    test_loader : DataLoader
        Dataloader containing the test dataset.
    device : torch.device
        'cuda' or 'cpu'.
    """
    model.eval()  # Switch to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            extra_features = batch['extra_features'].to(device)
            labels = batch['labels'].to(device)

            # Get model outputs (probabilities)
            outputs = model(input_ids, attention_mask, extra_features)

            # Convert probabilities to binary predictions using a threshold of 0.7
            predictions = (outputs > 0.7).float()

            # Compare with ground truth labels
            correct += (predictions == labels).sum().item()
            total += labels.numel()  # numel() gives the total number of elements

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")

###############################################################################
# 4. MAIN EXECUTION
###############################################################################
if __name__ == "__main__":
    # Decide if we have a GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the DNA-BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("zhihan1996/DNA_bert_6")

    # Create the dataset from the CSV file
    dataset = SpliceDataset("output.csv", tokenizer)
    # Get the number of extra features automatically
    extra_feature_dim = dataset.extra_feature_dim

    # Split the dataset into training and test sets (80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create DataLoader objects to batch and shuffle data
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the model with the dynamic extra_feature_dim
    model = SplicePredictor(extra_feature_dim=extra_feature_dim).to(device)

    # Use Binary Cross Entropy Loss for classification
    criterion = nn.BCELoss()

    # Use Adam optimizer for learning
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # If there is an existing checkpoint, load it
    checkpoint_path = "model_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading model checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Checkpoint loaded successfully!")

    # Train the model
    train_model(
        model, 
        train_loader, 
        criterion, 
        optimizer, 
        device, 
        num_epochs=10, 
        checkpoint_path=checkpoint_path
    )

    # Evaluate the model on the test set
    evaluate_model(model, test_loader, device)
