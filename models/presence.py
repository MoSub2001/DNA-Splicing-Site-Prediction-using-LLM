import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import random
import torchmetrics

# Settings
MODEL_NAME   = 'presence'
BATCH_SIZE   = 32
NUM_EPOCHS   = 6
LR           = 1e-4
CKPT_DIR     = os.path.join('model', f'{MODEL_NAME}_chunkSGD')
CACHE_DIR    = 'cached_data_with_probabilities'

os.makedirs(CKPT_DIR, exist_ok=True)

class ChunkDataset(Dataset):
    def __init__(self, path):
        obj = torch.load(path, map_location='cpu')
        self.embeddings = obj['embeddings']
        self.labels = obj['labels']

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class PresenceHead(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(768, embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        x = self.proj(emb)
        x = self.classifier(x)
        return x.mean(dim=1).squeeze(-1)

def evaluate_model(model, chunk_files, device, batch_size=32, threshold=0.5, writer=None, epoch=0):
    model.eval()
    all_preds = []
    all_labels = []

    precision = torchmetrics.Precision(task='binary', threshold=threshold).to(device)
    recall    = torchmetrics.Recall(task='binary', threshold=threshold).to(device)
    f1        = torchmetrics.F1Score(task='binary', threshold=threshold).to(device)
    accuracy  = torchmetrics.Accuracy(task='binary', threshold=threshold).to(device)
    confmat   = torchmetrics.ConfusionMatrix(task='binary', threshold=threshold, num_classes=2).to(device)

    with torch.no_grad():
        for chunk_file in chunk_files:
            dataset = ChunkDataset(chunk_file)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
            for embs, labels in loader:
                embs = embs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(embs)
                probs = torch.sigmoid(logits)
                preds = probs > threshold

                # Update metrics
                precision.update(probs, labels)
                recall.update(probs, labels)
                f1.update(probs, labels)
                accuracy.update(probs, labels)
                confmat.update(probs, labels)

                # For confusion matrix logging
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

    precision_score = precision.compute().item()
    recall_score    = recall.compute().item()
    f1_score        = f1.compute().item()
    acc_score       = accuracy.compute().item()
    cmatrix         = confmat.compute().cpu().numpy()

    if writer:
        writer.add_scalar("Val/Precision", precision_score, epoch)
        writer.add_scalar("Val/Recall", recall_score, epoch)
        writer.add_scalar("Val/F1", f1_score, epoch)
        writer.add_scalar("Val/Acc", acc_score, epoch)
        # Optional: writer.add_figure("Val/Confusion_Matrix", plot_confusion_matrix(cmatrix), epoch)

    print(f"Epoch {epoch} | Precision: {precision_score:.4f} | Recall: {recall_score:.4f} | F1: {f1_score:.4f} | Acc: {acc_score:.4f}")
    print(f"Confusion Matrix:\n{cmatrix}")

    # Reset metrics for the next epoch
    precision.reset()
    recall.reset()
    f1.reset()
    accuracy.reset()
    confmat.reset()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = PresenceHead().to(device)
    model = nn.DataParallel(base_model) if torch.cuda.device_count() > 1 else base_model
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=f"runs/{MODEL_NAME}_chunkSGD")

    # Optionally resume
    ckpts = sorted(f for f in os.listdir(CKPT_DIR) if f.endswith('_sd.pt'))
    start_epoch = 1
    if ckpts:
        state = torch.load(os.path.join(CKPT_DIR, ckpts[-1]), map_location=device)
        model.load_state_dict(state, strict=False)
        print("Resumed from", ckpts[-1], flush=True)

    # List all chunks across chromosomes and subfolders
    all_chunks = sorted(glob.glob(os.path.join(CACHE_DIR, "ch*/ch*_cached_chunk*.pt")))

    for epoch in trange(start_epoch, NUM_EPOCHS+1, desc="Epoch"):
        random.shuffle(all_chunks)
        epoch_loss = 0.0
        sample_count = 0

        for chunk_file in all_chunks:
            dataset = ChunkDataset(chunk_file)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
            model.train()
            chunk_bar = tqdm(loader, desc=f"Epoch {epoch} {os.path.basename(chunk_file)}", leave=False)
            for embs, labels in chunk_bar:
                embs = embs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                with autocast():
                    logits = model(embs)
                    loss = nn.BCEWithLogitsLoss()(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item() * embs.size(0)
                sample_count += embs.size(0)
                chunk_bar.set_postfix(loss=loss.item())
        avg_loss = epoch_loss / sample_count if sample_count else 0.0
        writer.add_scalar("Train/Loss", avg_loss, epoch)
        print(f"Epoch {epoch} done. Avg loss: {avg_loss:.6f}")

        # EVALUATION: loop over all chunks
        evaluate_model(model, all_chunks, device, batch_size=BATCH_SIZE, epoch=epoch, writer=writer)

        # Save checkpoint at each epoch
        ckpt = os.path.join(CKPT_DIR, f"epoch{epoch}_sd.pt")
        torch.save(model.state_dict(), ckpt)
        print(f"Saved checkpoint: {ckpt}", flush=True)

    writer.close()

if __name__ == '__main__':
    main()
