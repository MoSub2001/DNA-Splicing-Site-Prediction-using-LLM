
# import os
# import time
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# from transformers import AutoTokenizer, AutoModel, AutoConfig
# from torch.utils.data import Dataset, DataLoader
# import torch.nn.functional as F
# from tqdm import tqdm
# import numpy as np
# from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# # Import TensorBoard SummaryWriter
# from torch.utils.tensorboard import SummaryWriter

# from s4torch.layer import S4Layer



# class SpliceDataset(Dataset):
#     def __init__(self, csv_file=None, df=None, tokenizer=None, max_length=504, balance=False):
#         """
#         Either csv_file or df must be provided.
#         If balance is True, the dataset will be balanced on a per-chromosome basis (if a 'chromosome'
#         column is present) or overall otherwise.
#         """
#         if df is not None:
#             self.data = df.copy()
#         elif csv_file is not None:
#             self.data = pd.read_csv(csv_file)
#         else:
#             raise ValueError("Either csv_file or df must be provided.")

#         # Create a binary flag for exon presence: 1 if there is any exon information, 0 otherwise.
#         self.data['has_exon'] = self.data['[exon Indices]'].apply(
#             lambda s: 1 if len(str(s).strip()) > 0 and str(s).strip() != "[]" else 0
#         )
        
#         if balance:
#             if 'chromosome' in self.data.columns:
#                 # Balance each chromosome separately.
#                 balanced_list = []
#                 for chrom, group in self.data.groupby('chromosome'):
#                     pos_data = group[group['has_exon'] == 1]
#                     neg_data = group[group['has_exon'] == 0]
#                     min_count = min(len(pos_data), len(neg_data))
#                     if min_count > 0:
#                         pos_sample = pos_data.sample(min_count, random_state=42)
#                         neg_sample = neg_data.sample(min_count, random_state=42)
#                         balanced_group = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=42)
#                         balanced_list.append(balanced_group)
#                 self.data = pd.concat(balanced_list).reset_index(drop=True)
#             else:
#                 pos_data = self.data[self.data['has_exon'] == 1]
#                 neg_data = self.data[self.data['has_exon'] == 0]
#                 min_count = min(len(pos_data), len(neg_data))
#                 self.data = pd.concat([pos_data.sample(min_count, random_state=42),
#                                        neg_data.sample(min_count, random_state=42)]).sample(frac=1, random_state=42).reset_index(drop=True)
        
#         self.tokenizer = tokenizer
#         self.max_length = max_length  # Should be 504 tokens
#         self.sequences = self.data['Token']
#         self.exon_indices = self.data['[exon Indices]']
    
#     def __len__(self):
#         return len(self.sequences)
    
#     def __getitem__(self, idx):
#         sequence = self.sequences.iloc[idx]  # Raw 504-base sequence
#         exon_indices_str = self.exon_indices.iloc[idx]
        
#         # Create a binary target: 1 if there is any exon, else 0.
#         target = 1 if (len(str(exon_indices_str).strip()) > 0 and str(exon_indices_str).strip() != "[]") else 0
        
#         # Tokenize (no extra special tokens)
#                 # after
#         inputs = self.tokenizer(
#             sequence,
#             add_special_tokens=False,
#             return_tensors="pt",
#             padding="max_length",       # ← pad to max_length
#             truncation=True,
#             max_length=self.max_length
#         )

        
#         input_ids = inputs['input_ids'].squeeze(0)         # (504,)
#         attention_mask = inputs['attention_mask'].squeeze(0)  # (504,)

#         return {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#             'target': torch.tensor(target, dtype=torch.float)  # single scalar target
#         }


# class SplicePredictorHybrid(nn.Module):
#     def __init__(
#         self,
#         embed_model_name: str = "zhihan1996/DNABERT-2-117M",
#         conv_channels: int = 128,
#         ssm_dim: int = 128,
#         n_transformer_layers: int = 2,
#         n_heads: int = 4,
#         dropout: float = 0.2,
#         seq_length: int = 504,
#         pad_length: int = 512,  # must be >= seq_length and power of two
#     ):
#         super().__init__()

#         # 1) DNABERT embedder
#         cfg = AutoConfig.from_pretrained(embed_model_name, trust_remote_code=True)
#         cfg.use_flash_attention = False
#         self.bert = AutoModel.from_pretrained(embed_model_name, config=cfg, trust_remote_code=True)
#         # freeze all except last encoder + pooler
#         for n, p in self.bert.named_parameters():
#             if not (n.startswith("bert.encoder.layer.11") or n.startswith("bert.pooler")):
#                 p.requires_grad = False

#         # 2) Conv1d motif extractor
#         self.conv    = nn.Conv1d(cfg.hidden_size, conv_channels, kernel_size=8, padding=4)
#         self.relu    = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)

#         # 3) S4 layer (built for pad_length)
#         self.s4 = S4Layer(d_model=conv_channels, n=ssm_dim, l_max=pad_length)

#         # 4) Transformer encoder
#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=conv_channels,
#             nhead=n_heads,
#             dim_feedforward=conv_channels*2,
#             dropout=dropout,
#             batch_first=False
#         )
#         self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_transformer_layers)

#         # 5) Pool & classification head
#         self.pool = nn.AdaptiveMaxPool1d(1)
#         self.fc   = nn.Sequential(
#             nn.Linear(conv_channels, 64),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(64, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, input_ids, attention_mask=None):
#         # 1) BERT embeddings [B, L, H]
#         emb = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
#         # 2) Conv -> [B, C, L]
#         x = emb.permute(0,2,1)
#         x = self.relu(self.conv(x))
#         x = self.dropout(x)
#         # 3) S4 expects [L, B, C]
#         x = x.permute(2,0,1)        # [L_seq, B, C]

#         orig = x.size(0)            # 504
#         pad_amt = self.s4.l_max - orig  # 512-504=8
#         x = F.pad(x, (0,0,0,0,0,pad_amt))  # -> [512, B, C]

#         x = self.s4(x)              # [512, B, C]
#         x = x[:orig]                # back to [504, B, C]

#         # 4) Transformer ([L, B, C])
#         x = self.transformer(x)

#         # 5) Pool & head
#         x = x.permute(1,2,0)        # [B, C, L]
#         x = self.pool(x).squeeze(-1) # [B, C]
#         logits = self.fc(x).squeeze(-1)
#         return logits





# # ─── Evaluation and training functions remain the same, plus histogram logging ─
# def evaluate_with_confusion_matrix(model, loader, criterion, device, threshold=0.5):
#     model.eval()
#     total_loss = 0.0
#     all_preds = []
#     all_targets = []

#     with torch.no_grad():
#         for batch in loader:
#             input_ids = batch['input_ids'].to(device)
#             targets = batch['target'].to(device)  # shape: (batch,)
            
#             outputs = model(input_ids)  # shape: (batch, 1)
#             loss = criterion(outputs, targets.unsqueeze(1))
#             total_loss += loss.item()
            
#             preds = (outputs > threshold).float()  # shape: (batch, 1)
#             preds = preds.squeeze(1)  # shape: (batch,)
#             all_preds.extend(preds.cpu().numpy())
#             all_targets.extend(targets.cpu().numpy())
    
#     avg_loss = total_loss / len(loader)
#     all_preds = np.array(all_preds)
#     all_targets = np.array(all_targets)
    
#     cm = confusion_matrix(all_targets, all_preds)
#     accuracy = (cm.trace() / cm.sum()) if cm.sum() > 0 else 0.0
#     f1 = f1_score(all_targets, all_preds, zero_division=0)
#     precision = precision_score(all_targets, all_preds, zero_division=0)
#     recall = recall_score(all_targets, all_preds, zero_division=0)
    
#     # Kcorrect: count of sequences predicted correctly
#     k_correct = np.sum(all_preds == all_targets)
    
#     return avg_loss, accuracy, f1, precision, recall, cm, k_correct

# def train_model(model, train_loader, test_loader, criterion, optimizer, device,
#                 num_epochs=3, threshold=0.5, writer=None):
#     metrics_history = []
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
#             input_ids = batch['input_ids'].to(device)
#             mask = batch['attention_mask'].to(device)
#             targets = batch['target'].to(device)

#             optimizer.zero_grad()
#             outputs = model(input_ids, attention_mask=mask)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()

#         # log weight histograms
#         if writer:
#             for name, param in model.named_parameters():
#                 writer.add_histogram(name, param, epoch)

#         train_loss, train_acc, train_f1, train_precision, train_recall, train_cm, train_kcorrect = \
#             evaluate_with_confusion_matrix(model, train_loader, criterion, device, threshold)
#         # Evaluate on test set
#         test_loss, test_acc, test_f1, test_precision, test_recall, test_cm, test_kcorrect = \
#             evaluate_with_confusion_matrix(model, test_loader, criterion, device, threshold)
        
#         # Convert confusion matrices to strings for CSV logging
#         train_cm_str = str(train_cm.tolist())
#         test_cm_str = str(test_cm.tolist())
        
#         epoch_metrics = {
#             'epoch': epoch + 1,
#             'train_loss': train_loss,
#             'train_acc': train_acc,
#             'train_f1': train_f1,
#             'train_precision': train_precision,
#             'train_recall': train_recall,
#             'train_kcorrect': train_kcorrect,
#             'train_cm': train_cm_str,
#             'test_loss': test_loss,
#             'test_acc': test_acc,
#             'test_f1': test_f1,
#             'test_precision': test_precision,
#             'test_recall': test_recall,
#             'test_kcorrect': test_kcorrect,
#             'test_cm': test_cm_str
#         }
#         metrics_history.append(epoch_metrics)
        
#         # Log scalars to TensorBoard
#         if writer is not None:
#             writer.add_scalar("Loss/Train", train_loss, epoch)
#             writer.add_scalar("Loss/Test", test_loss, epoch)
#             writer.add_scalar("Accuracy/Train", train_acc, epoch)
#             writer.add_scalar("Accuracy/Test", test_acc, epoch)
#             writer.add_scalar("F1/Train", train_f1, epoch)
#             writer.add_scalar("F1/Test", test_f1, epoch)
#             writer.add_scalar("Precision/Train", train_precision, epoch)
#             writer.add_scalar("Precision/Test", test_precision, epoch)
#             writer.add_scalar("Recall/Train", train_recall, epoch)
#             writer.add_scalar("Recall/Test", test_recall, epoch)
#             writer.add_scalar("Kcorrect/Train", train_kcorrect, epoch)
#             writer.add_scalar("Kcorrect/Test", test_kcorrect, epoch)
    
    
#     return metrics_history


# # ─── Main execution: swap in SplicePredictorHybrid ─────────────────────────────
# if __name__ == "__main__":
#     print("starting ssm.py\n")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     writer = SummaryWriter(log_dir=f"runs/hybrid_ssm")

#     # Tokenizer & DataFrames as before…
#     tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
#     all_dfs = []
#     for chrom in range(1,5):
#         fp = os.path.join("../chromosomeData", f"ch{chrom}.csv")
#         if os.path.exists(fp) and os.path.getsize(fp)>0:
#             df = pd.read_csv(fp); df['chromosome']=chrom; all_dfs.append(df)
#     combined_df = pd.concat(all_dfs, ignore_index=True)
#     train_df = combined_df.sample(frac=0.8, random_state=42)
#     test_df  = combined_df.drop(train_df.index).reset_index(drop=True)

#     train_ds = SpliceDataset(df=train_df, tokenizer=tokenizer, max_length=504, balance=True)
#     test_ds  = SpliceDataset(df=test_df,  tokenizer=tokenizer, max_length=504, balance=True)
#     train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
#     test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

#     model = SplicePredictorHybrid().to(device)
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)

#     # (Optional) log graph
#     sample = next(iter(train_loader))
   

#     # Train
#     history = train_model(model, train_loader, test_loader,
#                           criterion, optimizer, device,
#                           num_epochs=15, threshold=0.5, writer=writer)

#     writer.close()
#     print("Done — TensorBoard logs in:", writer.log_dir)
#     print("Run: tensorboard --logdir runs/")



import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from s4torch.layer import S4Layer

class SplicePredictorHybrid(nn.Module):
    def __init__(
        self,
        embed_model_name: str = "zhihan1996/DNABERT-2-117M",
        conv_channels: int = 128,
        ssm_dim: int = 128,
        n_transformer_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.2,
        seq_length: int = 504,
    ):
        super().__init__()

        # 1) DNABERT embedder (frozen except last layer)
        cfg = AutoConfig.from_pretrained(embed_model_name, trust_remote_code=True)
        cfg.use_flash_attention = False
        self.bert = AutoModel.from_pretrained(embed_model_name, config=cfg, trust_remote_code=True, use_flash_attention=False)
        for name, param in self.bert.named_parameters():
            if not (name.startswith("bert.encoder.layer.11") or name.startswith("bert.pooler")):
                param.requires_grad = False

        # 2) Conv1d motif extractor
        self.conv = nn.Conv1d(cfg.hidden_size, conv_channels, kernel_size=8, padding=4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 3) S4 state‑space layer: use next power-of-two >= seq_length
        self.seq_length = seq_length
        self.l_max = 1 << (seq_length - 1).bit_length()  # e.g., 504 -> 512
        self.s4 = S4Layer(d_model=conv_channels, n=ssm_dim, l_max=self.l_max)

        # 4) Transformer encoder for feature mixing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conv_channels,
            nhead=n_heads,
            dim_feedforward=conv_channels*2,
            dropout=dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

        # 5) Pooling + classification head
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(conv_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask=None):
        # 1) BERT embeddings → [B, L, H]
        emb = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]

        # 2) Conv1D motif extraction → [B, C, L]
        x = emb.permute(0, 2, 1)
        x = self.relu(self.conv(x))
        x = self.dropout(x)

        # 3) Prepare for S4: [L, B, C]
        x = x.permute(2, 0, 1)
        orig = x.size(0)  # original sequence length (should be seq_length)

        #    Pad time dimension from seq_length → l_max
        pad_amt = self.l_max - orig
        x = F.pad(x, (0, 0, 0, 0, 0, pad_amt))  # pads the first dim (sequence)

        #    Single S4 call on length l_max
        x = self.s4(x)

        #    Crop back to original length
        x = x[:orig]

        # 4) Transformer encoder (expects [L, B, C])
        x = self.transformer(x)

        # 5) Pool & classification head
        x = x.permute(1, 2, 0)          # → [B, C, L]
        x = self.pool(x).squeeze(-1)    # → [B, C]
        logits = self.fc(x).squeeze(-1) # → [B]
        return logits


if __name__ == "__main__":
    # quick shape check
    B, L = 2, 504
    dummy_ids = torch.randint(0, 4, (B, L))
    dummy_mask = torch.ones(B, L, dtype=torch.long)
    model = SplicePredictorHybrid(seq_length=L)
    out = model(dummy_ids, attention_mask=dummy_mask)
    print("Output shape:", out.shape)  # should be [B]
