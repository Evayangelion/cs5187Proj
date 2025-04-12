# ==================== models/lightgcn.py (多负样本版 LightGCN) ====================
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, n_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, edge_index):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        all_emb = torch.cat([user_emb, item_emb], dim=0)

        embeddings = [all_emb]
        for _ in range(self.n_layers):
            all_emb = self.propagate(all_emb, edge_index)
            embeddings.append(all_emb)

        final_emb = torch.stack(embeddings, dim=0).mean(dim=0)
        return final_emb[:self.num_users], final_emb[self.num_users:]

    def propagate(self, x, edge_index):
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float().clamp(min=1)
        norm = torch.rsqrt(deg[row]) * torch.rsqrt(deg[col])
        out = torch.zeros_like(x)
        out.index_add_(0, row, x[col] * norm.unsqueeze(1))
        return out

class LightGCNRecommender:
    def __init__(self, num_users, num_items, embedding_dim=64, n_layers=3, lr=0.01, epochs=200, num_negs=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LightGCN(num_users, num_items, embedding_dim, n_layers).to(self.device)
        self.epochs = epochs
        self.num_negs = num_negs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.user_item_dict = defaultdict(set)

    def fit(self, train_data):
        user_map = {u: i for i, u in enumerate(set(u for u, _, _ in train_data))}
        item_map = {i: j for j, i in enumerate(set(i for _, i, _ in train_data))}
        self.user_map = user_map
        self.item_map = item_map
        self.inv_user_map = {i: u for u, i in user_map.items()}
        self.inv_item_map = {j: i for i, j in item_map.items()}

        num_user = len(user_map)
        num_item = len(item_map)

        edge_user = torch.tensor([user_map[u] for u, i, _ in train_data], dtype=torch.long)
        edge_item = torch.tensor([item_map[i] for u, i, _ in train_data], dtype=torch.long) + num_user
        self.edge_index = torch.cat([
            torch.stack([edge_user, edge_item], dim=0),
            torch.stack([edge_item, edge_user], dim=0)
        ], dim=1).to(self.device)

        for u, i, _ in train_data:
            if u in user_map and i in item_map:
                self.user_item_dict[user_map[u]].add(item_map[i])

        for epoch in range(self.epochs):
            self.model.train()
            user_emb, item_emb = self.model(self.edge_index)
            loss = self.bpr_loss(user_emb, item_emb, num_user, num_item)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def bpr_loss(self, user_emb, item_emb, num_users, num_items):
        batch_size = 1024
        u_idx = torch.randint(0, num_users, (batch_size,), device=self.device)
        i_idx = torch.tensor([random.choice(list(self.user_item_dict[u.item()])) for u in u_idx], device=self.device)

        loss_list = []
        for _ in range(self.num_negs):
            j_idx = torch.randint(0, num_items, (batch_size,), device=self.device)
            mask = torch.tensor([j.item() not in self.user_item_dict[u.item()] for u, j in zip(u_idx, j_idx)], device=self.device)
            if mask.sum() == 0:
                continue
            u = u_idx[mask]
            i = i_idx[mask]
            j = j_idx[mask]
            x_ui = (user_emb[u] * item_emb[i]).sum(dim=1)
            x_uj = (user_emb[u] * item_emb[j]).sum(dim=1)
            loss = -F.logsigmoid(x_ui - x_uj).mean()
            loss_list.append(loss)

        return torch.stack(loss_list).mean() if loss_list else torch.tensor(0.0, requires_grad=True, device=self.device)

    def recommend(self, user_id, all_items, k=10):
        if user_id not in self.user_map:
            return []
        self.model.eval()
        with torch.no_grad():
            user_emb, item_emb = self.model(self.edge_index)
            uid = self.user_map[user_id]
            scores = item_emb @ user_emb[uid]
            topk = torch.topk(scores, k).indices.cpu().numpy().tolist()
            return [self.inv_item_map[i] for i in topk if i in self.inv_item_map]
