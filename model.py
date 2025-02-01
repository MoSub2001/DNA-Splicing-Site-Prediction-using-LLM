import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np

# Define the dataset class
class SpliceSiteDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        # Filter sequences less than 500 in length
        self.data = self.data[self.data['Token'].apply(len) >= 500]

        # Process the features
        self.sequences = self.data['Token'].apply(self.sequence_to_tensor).values
        self.exon_presence = self.data['[exon # ]'].values > 0  # Binary target: 1 if exon exists, 0 otherwise
        
        # Features such as exon indices, intron indices, etc., can be processed further if needed

    def sequence_to_tensor(self, sequence):
        """Converts a DNA sequence to a numerical tensor."""
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}  # Example mapping
        return torch.tensor([mapping[base] for base in sequence if base in mapping], dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.sequences[idx], torch.tensor(self.exon_presence[idx], dtype=torch.float32)

# Define the neural network class
class SpliceSiteNN(nn.Module):
    def __init__(self):
        super(SpliceSiteNN, self).__init__()

        # Embedding layer to convert sequences to feature vectors
        self.embedding = nn.Embedding(4, 16)  # 4 unique bases, embedding size 16

        # Convolutional layers for sequence pattern recognition
        self.conv1 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)

        # Fully connected layers
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)  # Output layer

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # Rearrange for Conv1d (batch_size, channels, seq_length)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.mean(dim=2)  # Global average pooling
        x = self.relu(self.fc1(x))
        #x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# Load the dataset
csv_file = 'output2.csv'  # Replace with the actual path
batch_size = 32

dataset = SpliceSiteDataset(csv_file)

# Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model = SpliceSiteNN()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, data_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for sequences, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(data_loader):.4f}")

# Evaluate the model
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences, labels in data_loader:
            outputs = model(sequences)
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# Train the model
train(model, train_loader, criterion, optimizer, num_epochs=10)

# Evaluate on the test set
evaluate(model, test_loader)

# Save the model
torch.save(model.state_dict(), 'splice_site_model.pth')

print("Training complete and model saved.")
