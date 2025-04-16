# ==================== models/svd.py (增强版 SVD+BPR with 多负样本 + 正则项 + fallback 冷启动) ====================
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import logging
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

class SVDEmbedding(nn.Module):
    def __init__(self, num_users, num_items, emb_size=128):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, users, items):
        u = self.user_emb(users)
        i = self.item_emb(items)
        return (u * i).sum(dim=1)

class BPRDataset(Dataset):
    def __init__(self, user_item_pairs, num_users, num_items, num_neg=5):
        self.user_item = defaultdict(set)
        for u, i, _ in user_item_pairs:
            self.user_item[u].add(i)
        self.users = list(self.user_item.keys())
        self.num_users = num_users
        self.num_items = num_items
        self.num_neg = num_neg

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        pos_items = list(self.user_item[u])
        samples = []
        for _ in range(self.num_neg):
            i = random.choice(pos_items)
            j = random.randint(0, self.num_items - 1)
            while j in self.user_item[u]:
                j = random.randint(0, self.num_items - 1)
            samples.append((u, i, j))
        return samples

class SVDRecommender:
    def __init__(self, num_users, num_items, emb_size=512, lr=0.001, weight_decay=1e-4, epochs=200, batch_size=1024, num_neg=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SVDEmbedding(num_users, num_items, emb_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.0)  # weight_decay 手动加
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_users = num_users
        self.num_items = num_items
        self.num_neg = num_neg
        self.weight_decay = weight_decay
        self.popular_items = None

    def fit(self, train_data):
        user_map = {u: i for i, u in enumerate(sorted(set(u for u, _, _ in train_data)))}
        item_map = {i: j for j, i in enumerate(sorted(set(i for _, i, _ in train_data)))}
        self.user_map = user_map
        self.item_map = item_map
        self.inv_user_map = {i: u for u, i in user_map.items()}
        self.inv_item_map = {j: i for i, j in item_map.items()}

        mapped_data = [(user_map[u], item_map[i], 1.0) for u, i, _ in train_data if u in user_map and i in item_map]
        dataset = BPRDataset(mapped_data, len(user_map), len(item_map), num_neg=self.num_neg)

        # Flatten samples in __getitem__
        def collate_fn(batch):
            samples = [sample for user_samples in batch for sample in user_samples]
            u, i, j = zip(*samples)
            return torch.tensor(u), torch.tensor(i), torch.tensor(j)

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            start = time.time()
            for u, i, j in dataloader:
                u = u.to(self.device)
                i = i.to(self.device)
                j = j.to(self.device)

                x_ui = self.model(u, i)
                x_uj = self.model(u, j)

                reg_loss = (self.model.user_emb(u).norm(dim=1).pow(2).mean() +
                            self.model.item_emb(i).norm(dim=1).pow(2).mean() +
                            self.model.item_emb(j).norm(dim=1).pow(2).mean())

                loss = -F.logsigmoid(x_ui - x_uj).mean() + self.weight_decay * reg_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * u.size(0)

            logging.info(f"📘 SVD(BPR) Epoch {epoch + 1}/{self.epochs} 完成，Loss={total_loss/len(dataloader.dataset):.4f}，耗时={(time.time() - start):.2f}s")

        # Popularity fallback
        counter = defaultdict(int)
        for _, i, _ in train_data:
            counter[i] += 1
        self.popular_items = sorted(counter, key=counter.get, reverse=True)[:100]

    def recommend(self, user_id, all_items, k=10):
        if user_id not in self.user_map:
            return self.popular_items[:k]  # fallback for cold-start users

        self.model.eval()
        with torch.no_grad():
            uid = torch.tensor([self.user_map[user_id]], device=self.device)
            candidate_ids = [self.item_map[i] for i in all_items if i in self.item_map]
            items_tensor = torch.tensor(candidate_ids, device=self.device)
            uid_expand = uid.repeat(len(items_tensor))
            scores = self.model(uid_expand, items_tensor).cpu().numpy()
            top_k = torch.topk(torch.tensor(scores), k).indices.numpy()
            return [self.inv_item_map[candidate_ids[i]] for i in top_k]
