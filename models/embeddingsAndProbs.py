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
import gc
import torchmetrics

MODEL_NAME   = 'concat_softmax'
BATCH_SIZE   = 32
NUM_EPOCHS   = 4
LR           = 1e-4
CKPT_DIR     = os.path.join('model', MODEL_NAME)
CACHE_DIR    = 'cached_data_with_probabilities'
NUM_CLASSES  = 2

os.makedirs(CKPT_DIR, exist_ok=True)

class ConcatSegHead(nn.Module):
    def __init__(self, embed_dim=768, num_classes=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim + 1, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, emb, prob):
        B, L, D = emb.shape
        flat_emb  = emb.view(-1, D)
        flat_prob = prob.contiguous().view(-1, 1)
        out = self.head(torch.cat([flat_emb, flat_prob], dim=-1))  # [B*L, num_classes]
        return out.view(B, L, -1)  # [B, L, num_classes]

class ChunkSegDataset(Dataset):
    def __init__(self, path):
        obj = torch.load(path, map_location='cpu')
        self.embeddings = obj['embeddings']   # [N, L, D]
        self.probs      = obj['probs']        # [N, L]
        self.labels     = obj['labels']       # [N, L], float or int
        # Ensure correct type and shape
        assert self.labels.ndim == 2, f"labels.ndim={self.labels.ndim}, expected 2 ([N, L])"
    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self, idx):
        return self.embeddings[idx], self.probs[idx], torch.as_tensor(self.labels[idx], dtype=torch.long)

def train_one_epoch(model, chunk_files, optimizer, scaler, device, writer=None, epoch=0):
    model.train()
    total_loss, total_samples = 0.0, 0
    random.shuffle(chunk_files)
    for chunk_file in chunk_files:
        ds = ChunkSegDataset(chunk_file)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        for embs, probs, labels in tqdm(loader, desc=f"Train {os.path.basename(chunk_file)}", leave=False):
            embs = embs.to(device, non_blocking=True)
            probs = probs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # Debug shapes
            if epoch == 1:
                print(f"embs: {embs.shape}, probs: {probs.shape}, labels: {labels.shape}")
            optimizer.zero_grad()
            with autocast():
                logits = model(embs, probs)         # [B, L, num_classes]
                if epoch == 1:
                    print(f"logits: {logits.shape}, labels: {labels.shape}")
                # [B, L, num_classes] vs [B, L] → [B*L, num_classes] vs [B*L]
                loss = nn.CrossEntropyLoss()(logits.view(-1, NUM_CLASSES), labels.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * embs.size(0)
            total_samples += embs.size(0)
        del ds, loader
        torch.cuda.empty_cache(); gc.collect()
    avg_loss = total_loss / total_samples if total_samples else 0.0
    if writer: writer.add_scalar("Train/Loss", avg_loss, epoch)
    return avg_loss

def evaluate_seg_model(model, chunk_files, device, batch_size=32, writer=None, epoch=0):
    model.eval()
    f1 = torchmetrics.F1Score(task='multiclass', num_classes=NUM_CLASSES, average='macro').to(device)
    acc = torchmetrics.Accuracy(task='multiclass', num_classes=NUM_CLASSES, average='macro').to(device)
    prec = torchmetrics.Precision(task='multiclass', num_classes=NUM_CLASSES, average='macro').to(device)
    rec = torchmetrics.Recall(task='multiclass', num_classes=NUM_CLASSES, average='macro').to(device)
    confmat = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=NUM_CLASSES).to(device)

    with torch.no_grad():
        for chunk_file in chunk_files:
            ds = ChunkSegDataset(chunk_file)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
            for embs, probs, labels in loader:
                embs = embs.to(device, non_blocking=True)
                probs = probs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(embs, probs)   # [B, L, num_classes]
                pred_probs = torch.softmax(logits, dim=-1)
                pred_class = pred_probs.argmax(dim=-1)  # [B, L]
                # Flatten
                f1.update(pred_class.view(-1), labels.view(-1))
                acc.update(pred_class.view(-1), labels.view(-1))
                prec.update(pred_class.view(-1), labels.view(-1))
                rec.update(pred_class.view(-1), labels.view(-1))
                confmat.update(pred_class.view(-1), labels.view(-1))
            del ds, loader
            torch.cuda.empty_cache(); gc.collect()

    f1_score = f1.compute().item()
    acc_score = acc.compute().item()
    prec_score = prec.compute().item()
    rec_score = rec.compute().item()
    if writer:
        writer.add_scalar("Val/F1", f1_score, epoch)
        writer.add_scalar("Val/Acc", acc_score, epoch)
        writer.add_scalar("Val/Precision", prec_score, epoch)
        writer.add_scalar("Val/Recall", rec_score, epoch)
    print(f"Epoch {epoch} | F1: {f1_score:.4f} | Acc: {acc_score:.4f} | Prec: {prec_score:.4f} | Rec: {rec_score:.4f}")
    f1.reset(); acc.reset(); prec.reset(); rec.reset(); confmat.reset()

def main():
    print(f"[INFO] Training model: {MODEL_NAME}", flush=True)
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer    = SummaryWriter(log_dir=f'runs/{MODEL_NAME}')
    model     = ConcatSegHead(num_classes=NUM_CLASSES).to(device)  # Or MultiplySegHead
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler    = GradScaler()

    # Optionally resume weights
    ckpt_files = [f for f in os.listdir(CKPT_DIR) if f.endswith('_full.pt')]
    if ckpt_files:
        latest = sorted(ckpt_files)[-1]
        print(f"Resuming from {latest}", flush=True)
        model = torch.load(os.path.join(CKPT_DIR, latest)).to(device)

    # List all chunk files (search subfolders)
    all_chunks = sorted(glob.glob(os.path.join(CACHE_DIR, "ch*/ch*_cached_chunk*.pt")))
    print(f"[INFO] Found {len(all_chunks)} chunk files in {CACHE_DIR}")

    for epoch in trange(1, NUM_EPOCHS+1, desc="Epoch", unit="epoch"):
        train_one_epoch(model, all_chunks, optimizer, scaler, device, writer, epoch)
        evaluate_seg_model(model, all_chunks, device, batch_size=BATCH_SIZE, epoch=epoch, writer=writer)
        torch.save(model, os.path.join(CKPT_DIR, f'epoch{epoch}_full.pt'))
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, f'epoch{epoch}_sd.pt'))
        writer.flush()
        torch.cuda.empty_cache(); gc.collect()
    writer.close()

if __name__ == '__main__':
    main()
