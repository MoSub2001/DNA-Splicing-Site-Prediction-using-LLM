import os
import ast
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME    = 'concat'              # or 'multiply'
BATCH_SIZE    = 8
NUM_EPOCHS    = 4
LR            = 1e-4
CHROMOSOMES   = list(range(1,11))
PROB_DIR      = 'predictions'
MAX_LENGTH    = 504

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build per-pos boundary mask
# ─────────────────────────────────────────────────────────────────────────────
def create_target_vector(exon_indices_str, seq_len=MAX_LENGTH):
    vec = np.zeros(seq_len, dtype=np.float32)
    if isinstance(exon_indices_str, str) and exon_indices_str.startswith('['):
        intervals = ast.literal_eval(exon_indices_str)
        for iv in intervals:
            s = max(0, int(iv.get('start', 0)))
            e = min(seq_len-1, int(iv.get('end', seq_len-1)))
            vec[s] = 1.0
            vec[e] = 1.0
    return vec

# ─────────────────────────────────────────────────────────────────────────────
# Dataset: load one sequence’s embedding by inspecting its batch file
# ─────────────────────────────────────────────────────────────────────────────
class EmbeddingBatchDataset(Dataset):
    def __init__(self, batch_dir, df):
        paths = sorted([
            os.path.join(batch_dir, fn)
            for fn in os.listdir(batch_dir) if fn.endswith('.pt')
        ])
        sizes = [ torch.load(p).shape[0] for p in paths ]
        self.batch_paths = paths
        self.cum_sizes   = np.cumsum(sizes)
        self.probs       = df['prob'].tolist()
        self.counts      = df['[exon # ]'].astype(int).tolist()
        self.raw_indices = df['[exon Indices]'].tolist()

    def __len__(self):
        return len(self.probs)

    def __getitem__(self, idx):
        batch_i = np.searchsorted(self.cum_sizes, idx, side='right')
        offset  = idx - (self.cum_sizes[batch_i-1] if batch_i>0 else 0)
        emb_batch = torch.load(self.batch_paths[batch_i])
        emb       = emb_batch[offset]

        prob      = torch.tensor(self.probs[idx], dtype=torch.float)
        cnt       = torch.tensor(self.counts[idx], dtype=torch.long)
        mask_vec  = create_target_vector(self.raw_indices[idx])
        mask      = torch.tensor(mask_vec, dtype=torch.float)

        return {'emb': emb, 'prob': prob, 'target_mask': mask, 'target_count': cnt}

# ─────────────────────────────────────────────────────────────────────────────
# Models: small seg-head only
# ─────────────────────────────────────────────────────────────────────────────
class ConcatSegHead(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim + 1, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    def forward(self, emb, prob):
        B, L, D = emb.shape
        flat_emb  = emb.view(-1, D)
        flat_prob = prob.unsqueeze(1).repeat(1, L).view(-1,1)
        out = self.head(torch.cat([flat_emb, flat_prob], dim=-1))
        return out.view(B, L)

class MultiplySegHead(nn.Module):
    def __init__(self, embed_dim=768, nhead=8, conv_ch=64):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead),
            num_layers=1
        )
        self.conv1 = nn.Conv1d(embed_dim, conv_ch, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(0.2)
        self.fc    = nn.Linear(conv_ch, 1)
    def forward(self, emb, prob):
        B, L, D = emb.shape
        x = emb * prob.unsqueeze(-1)
        x = x.permute(1,0,2)
        x = self.transformer(x)
        x = x.permute(1,2,0)
        x = self.drop(self.relu(self.conv1(x)))
        x = x.permute(0,2,1)
        return self.fc(x).squeeze(-1)

MODEL_REGISTRY = {'concat': ConcatSegHead, 'multiply': MultiplySegHead}

# ─────────────────────────────────────────────────────────────────────────────
# Loss & loops
# ─────────────────────────────────────────────────────────────────────────────
pos_criterion = nn.BCEWithLogitsLoss()

def train_one_epoch(model, loader, optimizer, scaler, device, desc=None):
    model.train()
    total_loss, count = 0.0, 0
    for batch in tqdm(loader, desc=desc, leave=False):
        emb  = batch['emb'].to(device)
        prob = batch['prob'].to(device)
        tgt  = batch['target_mask'].to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(emb, prob)
            loss   = pos_criterion(logits.view(-1,1), tgt.view(-1,1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item(); count += 1
    return (total_loss/count) if count else 0.0

def evaluate(model, loader, device, desc=None):
    model.eval()
    all_P, all_T = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            emb   = batch['emb'].to(device)
            prob  = batch['prob'].to(device)
            tgt   = batch['target_mask'].to(device)
            cnts  = batch['target_count'].to(device)
            logits = model(emb, prob)
            probs  = torch.sigmoid(logits)
            pred   = torch.zeros_like(probs)
            for i, k in enumerate(cnts):
                topk = int(k)*2
                if topk>0:
                    idxs = torch.topk(probs[i], topk).indices
                    pred[i, idxs] = 1.0
            all_P.append(pred.view(-1).cpu()); all_T.append(tgt.view(-1).cpu())
    P = torch.cat(all_P).numpy(); T = torch.cat(all_T).numpy()
    return (accuracy_score(T,P), precision_score(T,P,zero_division=0), \
            recall_score(T,P,zero_division=0), f1_score(T,P,zero_division=0), \
            int(((P==1)&(T==1)).sum()))

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer    = SummaryWriter(log_dir=f'runs/{MODEL_NAME}')
    model     = MODEL_REGISTRY[MODEL_NAME]().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler    = GradScaler()

    for epoch in trange(1, NUM_EPOCHS+1, desc="Epoch", unit="epoch"):
        epoch_losses = []
        test_dfs     = []
        for c in CHROMOSOMES:
            dfc = pd.read_csv(f'../chromosomeData/ch{c}.csv')
            dfc['prob'] = torch.load(f'{PROB_DIR}/probvec_ch{c}.pt').tolist()
            dfc = dfc[dfc['Token'].str.len()>=MAX_LENGTH].reset_index(drop=True)
            if dfc.empty: continue
            train_df = dfc.sample(frac=0.8, random_state=42)
            test_df  = dfc.drop(train_df.index)
            test_dfs.append(test_df)
            pos = train_df[train_df['[exon # ]']>0]
            neg = train_df[train_df['[exon # ]']==0]
            if len(pos): neg = neg.sample(n=len(pos), random_state=42)
            train_df = pd.concat([pos,neg]).sample(frac=1, random_state=42)

            batch_dir    = f"bert_embeddings/batches_ch{c}"
            ds_train     = EmbeddingBatchDataset(batch_dir, train_df)
            train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
            loss = train_one_epoch(model, train_loader, optimizer, scaler, device,
                                   desc=f"Ch{c} train")
            epoch_losses.append(loss)

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        writer.add_scalar('Loss/train', avg_loss, epoch)

        if test_dfs:
            full_test = pd.concat(test_dfs, ignore_index=True)
            ds_test   = EmbeddingBatchDataset("bert_embeddings", full_test)
            test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE)
            acc,prec,rec,f1,tp = evaluate(model, test_loader, device,
                                          desc="Eval")
            writer.add_scalar('Seg/acc',     acc, epoch)
            writer.add_scalar('Seg/prec',    prec,epoch)
            writer.add_scalar('Seg/rec',     rec, epoch)
            writer.add_scalar('Seg/f1',      f1,   epoch)
            writer.add_scalar('Seg/true_pos',tp,   epoch)

        writer.flush()
        ckpt_dir = os.path.join('model', MODEL_NAME); os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f'epoch{epoch}.pt'))
        torch.cuda.empty_cache(); gc.collect()
    writer.close()

if __name__=='__main__':
    main()
