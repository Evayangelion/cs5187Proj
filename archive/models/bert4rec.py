# ==================== models/bert4rec_lightweight.py (CPU Friendly Optimized) ====================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
import logging
import time

class SlidingWindowBertDataset(Dataset):
    def __init__(self, user_sequences, max_len, mask_prob=0.2, num_items=0):
        self.samples = []
        self.mask_prob = mask_prob
        self.num_items = num_items
        for seq in user_sequences:
            for i in range(1, len(seq)):
                sub_seq = seq[max(0, i - max_len):i]
                target = seq[i]
                padded = [0] * (max_len - len(sub_seq)) + sub_seq
                self.samples.append((padded, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, label = self.samples[idx]
        masked = [
            self.num_items + 1 if t != 0 and random.random() < self.mask_prob else t
            for t in tokens
        ]
        return torch.tensor(masked), torch.tensor(label)

class BERT4Rec(nn.Module):
    def __init__(self, num_items, hidden_dim=64, num_heads=2, num_layers=1, max_len=50):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 2, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(hidden_dim, num_items + 1)
        self.register_buffer("positions", torch.arange(max_len).unsqueeze(0))
        self.max_len = max_len

    def forward(self, tokens):
        positions = self.positions.expand(tokens.size(0), -1)
        x = self.item_emb(tokens) + self.pos_emb(positions)
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.norm(x)
        return self.output(x)

class BERT4RecRecommender:
    def __init__(self, num_users, num_items, hidden_dim=64, max_len=50, lr=0.001, epochs=20, batch_size=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BERT4Rec(num_items, hidden_dim=hidden_dim, max_len=max_len).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.max_len = max_len
        self.num_items = num_items
        self.user_histories = defaultdict(list)

    def fit(self, train_data):
        for u, i, _ in train_data:
            self.user_histories[u].append(i)

        self.user_map = {u: i for i, u in enumerate(self.user_histories)}
        self.item_map = {i: j for j, i in enumerate(set(i for _, i, _ in train_data))}
        self.inv_item_map = {j: i for i, j in self.item_map.items()}

        sequences = [[self.item_map[i] for i in items if i in self.item_map] for items in self.user_histories.values()]
        dataset = SlidingWindowBertDataset(sequences, max_len=self.max_len, num_items=len(self.item_map))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            start = time.time()
            for tokens, labels in loader:
                tokens = tokens.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(tokens)[:, -1, :]
                loss = self.loss_fn(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * tokens.size(0)
            logging.info(f"📘 BERT4Rec Epoch {epoch+1}/{self.epochs} Loss={total_loss/len(dataset):.4f} ({time.time()-start:.2f}s)")

    def recommend(self, user_id, all_items, k=10):
        if user_id not in self.user_histories:
            return random.sample(all_items, k)
        self.model.eval()
        with torch.no_grad():
            seq = [self.item_map[i] for i in self.user_histories[user_id] if i in self.item_map]
            seq = seq[-self.max_len:]
            seq = [0] * (self.max_len - len(seq)) + seq
            tokens = torch.tensor(seq, device=self.device).unsqueeze(0)
            logits = self.model(tokens)[:, -1, :].squeeze()
            top_k = torch.topk(logits, k * 2).indices.cpu().numpy().tolist()
            recs = [self.inv_item_map[i] for i in top_k if i in self.inv_item_map and self.inv_item_map[i] not in self.user_histories[user_id]]
            return recs[:k]