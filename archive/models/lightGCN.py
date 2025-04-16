import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import logging
from collections import defaultdict


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, n_layers=2):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.n_layers = n_layers
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, edge_index):
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)

        for _ in range(self.n_layers):
            row, col = edge_index
            deg = torch.bincount(row, minlength=all_emb.size(0)).float()
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            messages = all_emb[col] * norm.unsqueeze(1)
            all_emb = all_emb.index_add(0, row, messages)  # 保持梯度追踪

        user_final, item_final = torch.split(all_emb, [self.user_emb.num_embeddings, self.item_emb.num_embeddings])
        return user_final, item_final


class LightGCNRecommender:
    def __init__(self, num_users, num_items, embedding_dim=256, n_layers=2, weight_decay=1e-4, lr=0.001, epochs=200):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LightGCN(num_users, num_items, embedding_dim, n_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.epochs = epochs
        self.num_users = num_users
        self.num_items = num_items

    def fit(self, train_data):
        user_map = {u: i for i, u in enumerate(set(u for u, _, _ in train_data))}
        item_map = {i: j for j, i in enumerate(set(i for _, i, _ in train_data))}
        self.user_map = user_map
        self.item_map = item_map
        self.inv_user_map = {i: u for u, i in user_map.items()}
        self.inv_item_map = {j: i for i, j in item_map.items()}

        self.user_item = defaultdict(set)
        edges = []
        for u, i, _ in train_data:
            if u in user_map and i in item_map:
                uid = user_map[u]
                iid = item_map[i]
                self.user_item[uid].add(iid)
                edges.append((uid, iid + self.num_users))

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = edge_index

        for epoch in range(self.epochs):
            start = time.time()
            self.model.train()
            user_emb, item_emb = self.model(edge_index)
            loss = self.vectorized_bpr_loss(user_emb, item_emb)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            end = time.time()
            logging.info(f"📘 LightGCN Epoch {epoch + 1}/{self.epochs} Loss={loss.item():.4f} Time={(end - start):.2f}s")

    def vectorized_bpr_loss(self, user_emb, item_emb, num_samples=2048, num_neg=5):
        users = random.sample(list(self.user_item.keys()), min(num_samples, len(self.user_item)))
        u_ids, i_ids, j_ids = [], [], []
        for u in users:
            pos_items = list(self.user_item[u])
            if not pos_items:
                continue
            for _ in range(num_neg):
                i = random.choice(pos_items)
                neg_pool = set(range(self.num_items)) - self.user_item[u]
                if not neg_pool:
                    continue
                j = random.choice(list(neg_pool))
                u_ids.append(u)
                i_ids.append(i)
                j_ids.append(j)

        if not u_ids:
            return torch.tensor(0.0, requires_grad=True, device=self.device)

        u = torch.tensor(u_ids, device=self.device)
        i = torch.tensor(i_ids, device=self.device)
        j = torch.tensor(j_ids, device=self.device)

        x_ui = (user_emb[u] * item_emb[i]).sum(dim=1)
        x_uj = (user_emb[u] * item_emb[j]).sum(dim=1)
        delta = x_ui - x_uj
        loss = -F.logsigmoid(delta).mean()

        reg_loss = 1e-4 * (
            user_emb[u].norm(2).pow(2) +
            item_emb[i].norm(2).pow(2) +
            item_emb[j].norm(2).pow(2)
        ) / len(u)

        return loss + reg_loss

    def recommend(self, user_id, all_items, k=10):
        if user_id not in self.user_map:
            return []
        self.model.eval()
        with torch.no_grad():
            user_emb, item_emb = self.model(self.edge_index)
            uid = self.user_map[user_id]
            u_e = user_emb[uid]
            item_ids = [i for i in all_items if i in self.item_map and self.item_map[i] not in self.user_item[uid]]
            if not item_ids:
                return []
            item_idxs = torch.tensor([self.item_map[i] for i in item_ids], device=self.device)
            i_e = item_emb[item_idxs]
            scores = torch.matmul(i_e, u_e)
            top_k = torch.topk(scores, k).indices.cpu().numpy()
            return [item_ids[i] for i in top_k]
