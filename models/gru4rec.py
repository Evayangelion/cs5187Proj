import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
import logging

class GRU4RecDataset(Dataset):
    def __init__(self, sequences, max_len):
        self.samples = []
        for seq in sequences:
            for i in range(1, len(seq)):
                input_seq = seq[max(0, i - max_len):i]
                target = seq[i]
                pad = [0] * (max_len - len(input_seq))
                self.samples.append((pad + input_seq, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

class GRU4Rec(nn.Module):
    def __init__(self, num_items, hidden_dim=128):
        super().__init__()
        self.emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, num_items + 1)

    def forward(self, x):
        x = self.emb(x)
        out, _ = self.gru(x)
        return self.output(out[:, -1])

class GRU4RecRecommender:
    def __init__(self, num_users, num_items, hidden_dim=128, max_len=50, lr=0.001, epochs=20, batch_size=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.user_histories = defaultdict(list)

        self.item_map = {}
        self.inv_item_map = {}
        self.num_items = 0  # Will be set later dynamically

    def fit(self, train_data):
        for u, i, _ in train_data:
            self.user_histories[u].append(i)

        unique_items = sorted(set(i for seq in self.user_histories.values() for i in seq))
        self.item_map = {i: idx + 1 for idx, i in enumerate(unique_items)}  # +1 to reserve 0 as padding
        self.inv_item_map = {v: k for k, v in self.item_map.items()}
        self.num_items = len(self.item_map)

        self.model = GRU4Rec(self.num_items, hidden_dim=self.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        sequences = [[self.item_map[i] for i in items if i in self.item_map] for items in self.user_histories.values() if len(items) >= 2]
        dataset = GRU4RecDataset(sequences, self.max_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.loss_fn(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * x.size(0)
            logging.info(f"\U0001F4D8 GRU4Rec Epoch {epoch + 1}/{self.epochs} Loss={total_loss / len(dataset):.4f}")

    def recommend(self, user_id, all_items, k=10):
        if user_id not in self.user_histories or len(self.user_histories[user_id]) < 1:
            return random.sample(all_items, k)

        self.model.eval()
        with torch.no_grad():
            seq = [self.item_map[i] for i in self.user_histories[user_id] if i in self.item_map][-self.max_len:]
            seq = [0] * (self.max_len - len(seq)) + seq
            x = torch.tensor([seq], device=self.device)
            logits = self.model(x).squeeze()
            top_k = torch.topk(logits, k * 2).indices.tolist()
            recs = [self.inv_item_map[i] for i in top_k if i in self.inv_item_map and self.inv_item_map[i] not in self.user_histories[user_id]]
            return recs[:k]