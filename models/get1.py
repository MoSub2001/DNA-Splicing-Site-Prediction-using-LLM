#!/usr/bin/env python3
import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.serialization import safe_globals

# ─── 1) Register HF remote code so transformers_modules.zhihan1996 exists ───────
_ = AutoConfig.from_pretrained('zhihan1996/DNABERT-2-117M', trust_remote_code=True)
_ = AutoModel.from_pretrained('zhihan1996/DNABERT-2-117M', trust_remote_code=True)

# ─── 2) Import your custom classes so unpickling can find them ──────────────────
from embedingsandprobs import ConcatMultiHead, MultiplyMultiHead, MODEL_REGISTRY, SpliceDataset

# ─── 3) Configuration ──────────────────────────────────────────────────────────
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME   = 'concat'                   # or 'multiply'
CKPT_PATH    = f'model/{MODEL_NAME}/epoch_7.pt'
PROB_DIR     = 'predictions'
BATCH_SIZE   = 64
CHROMOSOMES  = list(range(1,23))
MAX_LENGTH   = 504
LAMBDA_COUNT = 1.0

# ─── 4) Prepare TensorBoard writer ─────────────────────────────────────────────
writer = SummaryWriter(log_dir=f"runs/{MODEL_NAME}_eval")

# ─── 5) Criteria ────────────────────────────────────────────────────────────────
pos_criterion   = torch.nn.BCEWithLogitsLoss()
count_criterion = torch.nn.CrossEntropyLoss()

# ─── 6) Load & unpickle your full-model checkpoint ─────────────────────────────
with safe_globals([ConcatMultiHead, MultiplyMultiHead]):
    model = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
model.eval()

# ─── 7) Tokenizer ──────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained('zhihan1996/DNABERT-2-117M', trust_remote_code=True)

# ─── 8) Accumulators ───────────────────────────────────────────────────────────
true_positives  = 0
total_predicted = 0
total_actual    = 0

count_preds = []
count_tgts  = []
P_list      = []
T_list      = []

total_loss  = 0.0
num_batches = 0

# ─── 9) Iterate & evaluate ─────────────────────────────────────────────────────
with torch.no_grad():
    for c in CHROMOSOMES:
        csv_path  = os.path.join('../chromosomeData', f'ch{c}.csv')
        prob_path = os.path.join(PROB_DIR,         f'probvec_ch{c}.pt')
        if not (os.path.exists(csv_path) and os.path.exists(prob_path)):
            continue

        df = pd.read_csv(csv_path)
        df = df[df['Token'].str.len() >= MAX_LENGTH].reset_index(drop=True)
        if df.empty:
            continue

        # inject prob column exactly as in training
        prob_arr  = torch.load(prob_path, map_location='cpu')
        prob_list = prob_arr.tolist() if isinstance(prob_arr, torch.Tensor) else prob_arr
        n         = min(len(df), len(prob_list))
        df        = df.iloc[:n].copy()
        df['prob'] = prob_list[:n]

        ds     = SpliceDataset(df, tokenizer)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

        for batch in loader:
            inp = {
                'input_ids':      batch['input_ids'].to(DEVICE),
                'attention_mask': batch['attention_mask'].to(DEVICE),
                'prob':           batch['prob'].to(DEVICE),
            }
            seg_logits, cnt_logits = model(**inp)
            seg_probs              = torch.sigmoid(seg_logits)       # (B, L)

            # compute losses
            seg_tgt = batch['target_mask'].to(DEVICE)
            cnt_tgt = batch['target_count'].to(DEVICE)
            pos_loss = pos_criterion(seg_logits.view(-1,1), seg_tgt.view(-1,1))
            cnt_loss = count_criterion(cnt_logits, cnt_tgt)
            loss     = pos_loss + LAMBDA_COUNT * cnt_loss
            total_loss  += loss.item()
            num_batches += 1

            # rebuild 0/1 mask_pred
            k_pred    = torch.argmax(cnt_logits, dim=1)             # (#ones per sample)
            mask_pred = torch.zeros_like(seg_probs)
            for i, k in enumerate(k_pred):
                if k > 0:
                    topk_idx       = torch.topk(seg_probs[i], k).indices
                    mask_pred[i, topk_idx] = 1.0

            # accumulate segmentation arrays for metrics
            P_list.append(mask_pred.cpu().view(-1).numpy())
            T_list.append(seg_tgt.cpu().view(-1).numpy())

            # accumulate count predictions
            count_preds.append(k_pred.cpu().numpy())
            count_tgts.append(cnt_tgt.cpu().numpy())

            # also count true positives etc.
            true_positives  += ((mask_pred == 1) & (seg_tgt == 1)).sum().item()
            total_predicted += mask_pred.sum().item()
            total_actual    += seg_tgt.sum().item()

# ─── 10) Final metrics ─────────────────────────────────────────────────────────
P = np.concatenate(P_list)
T = np.concatenate(T_list)
seg_acc   = accuracy_score(T, P)
seg_prec  = precision_score(T, P, zero_division=0)
seg_rec   = recall_score(T, P, zero_division=0)
seg_f1    = f1_score(T, P, zero_division=0)

Cp = np.concatenate(count_preds)
Ct = np.concatenate(count_tgts)
cnt_acc = (Cp == Ct).mean()

avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')

# ─── 11) Log to TensorBoard ────────────────────────────────────────────────────
step = 0
writer.add_scalar("eval/loss",          avg_loss,   step)
writer.add_scalar("eval/seg_accuracy",  seg_acc,    step)
writer.add_scalar("eval/seg_precision", seg_prec,   step)
writer.add_scalar("eval/seg_recall",    seg_rec,    step)
writer.add_scalar("eval/seg_f1",        seg_f1,     step)
writer.add_scalar("eval/count_accuracy", cnt_acc,   step)
writer.close()

# ─── 12) Print summary ─────────────────────────────────────────────────────────
print(f"✅ Eval loss              : {avg_loss:.4f}")
print(f"✅ Segmentation Acc       : {seg_acc:.4f}")
print(f"✅ Segmentation Prec      : {seg_prec:.4f}")
print(f"✅ Segmentation Recall    : {seg_rec:.4f}")
print(f"✅ Segmentation F1        : {seg_f1:.4f}")
print(f"✅ Count head Accuracy    : {cnt_acc:.4f}")
print(f"ℹ️  True positives (1’s)   : {true_positives}")
print(f"ℹ️  Total 1’s predicted   : {total_predicted}")
print(f"ℹ️  Total 1’s actual      : {total_actual}")
