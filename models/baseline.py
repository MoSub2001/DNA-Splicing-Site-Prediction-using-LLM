import os
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.amp import autocast, GradScaler

# Map nucleotides to integers
NT_MAP = {'A': 1, 'C': 2, 'G': 3, 'T': 4}

def safe_parse(x):
    """Convert a stringified list or space-separated numbers to a Python list of floats."""
    if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
        inner = x[1:-1].strip()
        if ',' in inner:
            parts = [p.strip() for p in inner.split(',') if p.strip()]
            try:
                return [float(p) for p in parts]
            except:
                pass
        parts = inner.split()
        try:
            return [float(p) for p in parts]
        except:
            pass
    return x

def preprocess_feature(v, max_length):
    """
    Turn a scalar or list/array of floats into a length-max_length ndarray,
    padding or truncating as needed, with NaNs/infs -> 0.
    """
    if isinstance(v, (int, float)):
        arr = np.full(max_length, v, dtype=float)
    else:
        arr = np.array(v, dtype=float)
        if arr.ndim == 0:
            arr = np.full(max_length, float(arr), dtype=float)
        if arr.shape[0] < max_length:
            pad = np.zeros(max_length - arr.shape[0], dtype=float)
            arr = np.concatenate([arr, pad], axis=0)
        else:
            arr = arr[:max_length]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

class SpliceIterableDataset(IterableDataset):
    """
    Streams ch1.csv, ch10.csv, ch11.csv from data_dir,
    chunk-wise loading with on-the-fly balancing using [exon # ].
    """
    def __init__(self, data_dir, max_length=504, chunksize=5000):
        super().__init__()
        self.data_dir = data_dir
        self.max_length = max_length
        self.chunksize = chunksize
        self.files = ['ch1.csv', 'ch10.csv', 'ch11.csv']
        # Determine feature columns
        sample = pd.read_csv(os.path.join(data_dir, self.files[0]), nrows=100)
        for col in sample.columns:
            if sample[col].dtype == object:
                sample[col] = sample[col].apply(safe_parse)
        exclude = lambda c: 'Indices' in c or c == 'Token'
        def is_valid(x):
            return isinstance(x, (int, float)) or (
                isinstance(x, (list, np.ndarray)) and len(x) > 0
                and all(isinstance(e, (int, float)) for e in x)
            )
        self.feature_cols = [c for c in sample.columns if not exclude(c) and sample[c].map(is_valid).all()]
        self.char2id = NT_MAP.copy()

    def _make_label(self, intervals):
        lbl = np.zeros(self.max_length, dtype=np.float32)
        if isinstance(intervals, list):
            for iv in intervals:
                s = max(0, int(iv.get('start', 0)))
                e = min(self.max_length - 1, int(iv.get('end', self.max_length - 1)))
                lbl[s] = 1
                lbl[e] = 1
        return torch.tensor(lbl)

    def __iter__(self):
        for fname in self.files:
            path = os.path.join(self.data_dir, fname)
            for chunk in pd.read_csv(path, chunksize=self.chunksize):
                # Filter exact length sequences
                chunk = chunk[chunk['Token'].str.len() == self.max_length]
                # Parse object columns
                for col in chunk.columns:
                    if chunk[col].dtype == object:
                        chunk[col] = chunk[col].apply(safe_parse)
                # Balance positives/negatives
                pos = chunk[chunk['[exon # ]'] > 0]
                neg = chunk[chunk['[exon # ]'] == 0]
                if len(pos) == 0:
                    continue
                neg_sample = neg.sample(n=len(pos), random_state=42)
                balanced = pd.concat([pos, neg_sample]).sample(frac=1, random_state=42)
                for _, row in balanced.iterrows():
                    seq = row['Token']
                    ids = [self.char2id.get(ch, 0) for ch in seq]
                    x = torch.tensor(ids, dtype=torch.long)
                    mats = [preprocess_feature(row[c], self.max_length) for c in self.feature_cols]
                    feats = torch.tensor(np.stack(mats, axis=1), dtype=torch.float32)
                    lbl = self._make_label(row['[exon Indices]'])
                    yield x, feats, lbl

class SplicePredictor(nn.Module):
    def __init__(self, embed_dim=16, feature_dim=0, lstm_hidden=64):
        super().__init__()
        self.embedding = nn.Embedding(5, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim + feature_dim, lstm_hidden,
                            batch_first=True, bidirectional=True)
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128), nn.ReLU(),
            nn.Linear(128, 64),           nn.ReLU(),
            nn.Linear(64, 32),            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, seq, feats):
        emb = self.embedding(seq)
        x = torch.cat([emb, feats], dim=-1)
        out, _ = self.lstm(x)
        return self.mlp(out).squeeze(-1)

if __name__ == '__main__':
    print("Starting balanced training to 'model/baseline'")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler()

    # Dataset & DataLoader
    data_dir = "with_pyrimidine"
    ds = SpliceIterableDataset(data_dir, max_length=504, chunksize=5000)
    dl = DataLoader(ds, batch_size=4, num_workers=0, pin_memory=False)

    # Setup
    writer = SummaryWriter(log_dir='runs/baseline')
    exon_idx = ds.feature_cols.index('[exon # ]')
    model = SplicePredictor(feature_dim=len(ds.feature_cols)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Ensure model base dir
    base_dir = 'model'
    os.makedirs(base_dir, exist_ok=True)

    epochs = 10
    for ep in range(1, epochs+1):
        total_loss = num_batches = 0
        TP = FP = TN = FN = 0
        for seq, feats, lbl in tqdm(dl, desc=f'Epoch {ep}'):
            # Move all inputs to the correct device
            seq, feats, lbl = seq.to(device), feats.to(device), lbl.to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                logits = model(seq, feats)
                loss = criterion(logits, lbl)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            num_batches += 1

            with torch.no_grad():
                sm = torch.softmax(logits, dim=1)
                preds = torch.zeros_like(sm)
                counts = feats[:,0,exon_idx].long()
                for i,c in enumerate(counts):
                    k = int(2*c.item())
                    if k>0:
                        idxs = torch.topk(sm[i],k).indices
                        preds[i,idxs]=1
                flat_p = preds.view(-1)
                flat_l = lbl.view(-1)
                TP += ((flat_p==1)&(flat_l==1)).sum().item()
                FP += ((flat_p==1)&(flat_l==0)).sum().item()
                FN += ((flat_p==0)&(flat_l==1)).sum().item()
                TN += ((flat_p==0)&(flat_l==0)).sum().item()

        # Compute metrics
        avg_loss = total_loss / num_batches if num_batches>0 else 0
        total = TP+FP+TN+FN
        acc = (TP+TN)/total if total>0 else 0
        prec = TP/(TP+FP) if TP+FP>0 else 0
        rec = TP/(TP+FN) if TP+FN>0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0

        # Log only specified metrics
        writer.add_scalar('Loss/train_epoch', avg_loss, ep)
        writer.add_scalar('Metrics/Accuracy',  acc,  ep)
        writer.add_scalar('Metrics/Precision', prec, ep)
        writer.add_scalar('Metrics/Recall',    rec,  ep)
        writer.add_scalar('Metrics/F1',        f1,   ep)
        writer.add_scalar('Metrics/TP',        TP,   ep)

        # Save per-epoch
        epoch_dir = os.path.join(base_dir, f'epoch_{ep}')
        os.makedirs(epoch_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(epoch_dir,'state.pth'))
        torch.save(model,             os.path.join(epoch_dir,'model.pth'))

    # Final model
    final_dir = os.path.join(base_dir,'final')
    os.makedirs(final_dir, exist_ok=True)
    torch.save(model, os.path.join(final_dir,'model.pth'))
    writer.close()
    print(f"Training complete. Models saved under '{base_dir}/baseline' ")
