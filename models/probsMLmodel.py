import os  # for filesystem path operations
import ast  # to safely evaluate string representations of Python literals
import torch  # main PyTorch library for tensors and models
import torch.nn as nn  # neural network modules
from torch.utils.data import Dataset, DataLoader  # dataset abstraction and data loading
from torch.utils.tensorboard import SummaryWriter  # logging metrics to TensorBoard
import pandas as pd  # data manipulation and CSV reading
import numpy as np  # numerical operations on arrays
from tqdm import tqdm  # progress bar for loops
from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix  # evaluation metrics

# ─────────────────────────────────────────────────────────────────────────────
# 1. Create label vector from exon indices
# ─────────────────────────────────────────────────────────────────────────────
def create_target_vector(exon_indices_str, sequence_length=504):
    # initialize a zero vector of given length (504 by default)
    label = np.zeros(sequence_length, dtype=np.float32)

    # if input is not a string, return all-zero tensor immediately
    if not isinstance(exon_indices_str, str):
        return torch.tensor(label)

    try:
        # parse the string representation of intervals into Python list of dicts
        intervals = ast.literal_eval(exon_indices_str)
        for interval in intervals:
            # ensure start index is at least 0
            s = max(0, int(interval['start']))
            # ensure end index does not exceed sequence length-1
            e = min(sequence_length - 1, int(interval['end']))
            # mark the start and end positions in label as 1
            label[s] = 1
            label[e] = 1
    except Exception as e:
        # print a warning if parsing fails
        print(f"⚠️ Failed to parse exon indices: {e}")

    # convert numpy array to torch tensor and return
    return torch.tensor(label)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset Class with padding workaround
# ─────────────────────────────────────────────────────────────────────────────
class ChromosomeDataset(Dataset):
    def __init__(self, chrom, probvec_dir, csv_dir, sequence_length=504):
        # build file paths for probability vectors and CSV data
        prob_path = os.path.join(probvec_dir, f"probvec_ch{chrom}.pt")
        csv_path  = os.path.join(csv_dir,   f"ch{chrom}.csv")

        # if either file missing, set empty dataset
        if not os.path.exists(prob_path) or not os.path.exists(csv_path):
            self.X = torch.empty(0)
            self.y = torch.empty(0)
            return

        # load saved tensor of shape (M, L): M windows, L-length vector each
        prob_tensor = torch.load(prob_path)  # (M, L)
        # read metadata CSV with exon indices per row
        df = pd.read_csv(csv_path)
        M, L = prob_tensor.size()
        P    = len(df)  # number of rows/windows expected

        # pad (if fewer prob entries) or truncate tensor to match CSV rows
        if M < P:
            prob_full = torch.zeros((P, L), dtype=prob_tensor.dtype)
            prob_full[:M] = prob_tensor
        else:
            prob_full = prob_tensor[:P]

        xs, ys = [], []
        # iterate through each window
        for i in range(P):
            x = prob_full[i]  # probability vector for window i
            # create binary label vector for window i
            y = create_target_vector(df.iloc[i]['[exon Indices]'], sequence_length=L)
            # only include examples with correct label shape
            if y.shape == (L,):
                xs.append(x)
                ys.append(y)

        # stack collected examples or leave empty
        self.X = torch.stack(xs) if xs else torch.empty(0)
        self.y = torch.stack(ys) if ys else torch.empty(0)

    def __len__(self):
        # number of examples in dataset
        return len(self.X)

    def __getitem__(self, idx):
        # return one input-target pair
        return self.X[idx], self.y[idx]

# ─────────────────────────────────────────────────────────────────────────────
# 3. Model definition: simple MLP for splice-site prediction
# ─────────────────────────────────────────────────────────────────────────────
class SpliceSiteMLP(nn.Module):
    def __init__(self, input_dim=504):
        super().__init__()
        # define a three-layer fully connected network
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),  # project to 256 dims
            nn.ReLU(),                   # activation
            nn.Dropout(0.2),             # regularization
            nn.Linear(256, 128),         # project to 128 dims
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, input_dim)    # project back to sequence length dim
        )
    def forward(self, x):
        # apply network and sigmoid to get probabilities in [0,1]
        return torch.sigmoid(self.model(x))  # output shape: (B, 504)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluation helper: compute metrics over dataset
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, loader, device):
    model.eval()  # set model to evaluation mode
    all_preds = []
    all_labels = []
    with torch.no_grad():  # disable gradient computation
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)  # get predicted probabilities
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    # flatten lists of tensors to 1D vectors
    all_preds  = torch.cat(all_preds).view(-1)
    all_labels = torch.cat(all_labels).view(-1)
    # threshold at 0.5 to get binary predictions
    pred_labels = (all_preds > 0.5).float()

    # compute true positives, true negatives, false positives, false negatives
    TP = ((pred_labels == 1) & (all_labels == 1)).sum().item()
    TN = ((pred_labels == 0) & (all_labels == 0)).sum().item()
    FP = ((pred_labels == 1) & (all_labels == 0)).sum().item()
    FN = ((pred_labels == 0) & (all_labels == 1)).sum().item()

    # calculate accuracy, recall, precision, F1, with epsilon to avoid div by zero
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    recall   = TP / (TP + FN + 1e-8)
    precision= TP / (TP + FP + 1e-8)
    f1       = 2 * precision * recall / (precision + recall + 1e-8)

    return accuracy, recall, f1, TP  # return key metrics

# ─────────────────────────────────────────────────────────────────────────────
# 5. Training loop: one epoch over data loader
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, loader, optimizer, criterion, device):
    model.train()  # set model to training mode
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()      # reset gradients
        pred = model(X)            # forward pass
        loss = criterion(pred, y)  # compute loss (BCELoss)
        loss.backward()            # backpropagate gradients
        optimizer.step()           # update weights
        total_loss += loss.item()  # accumulate loss value
    return total_loss / len(loader)  # return average loss per batch

# ─────────────────────────────────────────────────────────────────────────────
# 6. Main: orchestrate dataset loading, training, evaluation, logging
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # choose GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # instantiate model and move to device
    model = SpliceSiteMLP().to(device)
    # binary cross-entropy loss
    criterion = nn.BCELoss()
    # Adam optimizer with learning rate 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # TensorBoard writer to log metrics
    writer = SummaryWriter(log_dir="runs/per_chrom")

    num_epochs = 25
    for epoch in range(1, num_epochs + 1):
        print(f"\n=== 🌱 Epoch {epoch} ===")
        epoch_loss = 0.0
        chrom_count = 0

        # iterate through chromosomes 1 to 22
        for chrom in range(1, 23):
            print(f"  → Chromosome {chrom}", end="", flush=True)
            # load dataset for this chromosome
            ds = ChromosomeDataset(chrom, probvec_dir="predictions", csv_dir="../chromosomeData")
            if len(ds) == 0:
                print(": no data, skipping.")
                continue

            # create data loader with batch size 64
            loader = DataLoader(ds, batch_size=64, shuffle=True)
            # train one epoch on this chromosome
            loss = train_model(model, loader, optimizer, criterion, device)
            print(f": trained on {len(ds)} samples, loss={loss:.4f}")

            # log loss for this chromosome
            writer.add_scalar(f"Loss/Chrom{chrom}", loss, epoch)

            # evaluate on same data and log metrics
            acc, recall, f1, true_ones = evaluate_model(model, loader, device)
            writer.add_scalar(f"Accuracy/Chrom{chrom}", acc, epoch)
            writer.add_scalar(f"Recall/Chrom{chrom}", recall, epoch)
            writer.add_scalar(f"F1/Chrom{chrom}", f1, epoch)
            writer.add_scalar(f"TruePositives/Chrom{chrom}", true_ones, epoch)

            epoch_loss += loss
            chrom_count += 1

            # clean up to free GPU memory
            del ds, loader
            torch.cuda.empty_cache()

        # compute and log average loss across chromosomes
        avg_epoch_loss = epoch_loss / max(1, chrom_count)
        print(f"⭐ Epoch {epoch} average loss: {avg_epoch_loss:.4f}")
        writer.add_scalar("Loss/Average", avg_epoch_loss, epoch)

    # save trained model weights
    torch.save(model.state_dict(), "splice_model_per_ch.pt")
    # close TensorBoard writer
    writer.close()
    print("✅ Done. Model weights and metrics saved.")

if __name__ == "__main__":
    main()  # run main when script is executed directly
