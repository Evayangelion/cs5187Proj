import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import time
import logging
import random
from torch.utils.data import Dataset, DataLoader

class SVDPPEmbedding(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.implicit_item_emb = nn.Embedding(num_items, emb_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        nn.init.normal_(self.implicit_item_emb.weight, std=0.1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids, item_ids, offsets, flat_implicit):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)

        # 处理隐式反馈
        imp_emb = self.implicit_item_emb(flat_implicit)
        sizes = torch.diff(torch.cat([offsets, torch.tensor([len(flat_implicit)], device=offsets.device)])).tolist()
        packed = torch.split(imp_emb, sizes)
        summed = torch.stack([
            p.sum(0) / torch.sqrt(torch.tensor(p.size(0), dtype=torch.float, device=p.device))
            if p.size(0) > 0 else torch.zeros_like(u[0])
            for p in packed
        ])

        final_user = u + summed

        dot = (final_user * i).sum(dim=1)
        pred = dot + self.user_bias(user_ids).squeeze() + self.item_bias(item_ids).squeeze() + self.global_bias
        return pred

class SVDPPRecommender:
    def __init__(self, num_users, num_items, emb_dim=64, lr=0.005, weight_decay=0.0, epochs=50, batch_size=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SVDPPEmbedding(num_users, num_items, emb_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_users = num_users
        self.num_items = num_items

    def fit(self, train_data):
        best_loss = float('inf')
        patience = 5  # 连续多少轮没有提升就 early stop
        no_improve = 0

        self.user_map = {u: i for i, u in enumerate(set(u for u, _, _ in train_data))}
        self.item_map = {i: j for j, i in enumerate(set(i for _, i, _ in train_data))}
        self.inv_user_map = {i: u for u, i in self.user_map.items()}
        self.inv_item_map = {j: i for i, j in self.item_map.items()}

        # 用户历史
        user_hist = defaultdict(list)
        for u, i, _ in train_data:
            if u in self.user_map and i in self.item_map:
                user_hist[self.user_map[u]].append(self.item_map[i])

        self.user_hist = user_hist

        train_samples = [
            (self.user_map[u], self.item_map[i], r)
            for u, i, r in train_data if u in self.user_map and i in self.item_map
        ]

        dataset = [(u, i, r, user_hist[u]) for u, i, r in train_samples]
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self._collate_fn)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            for u, i, r, offsets, flat in loader:
                u, i, r = u.to(self.device), i.to(self.device), r.to(self.device)
                offsets = offsets.to(self.device)
                flat = flat.to(self.device)

                pred = self.model(u, i, offsets, flat)
                loss = F.mse_loss(pred, r)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(u)

                        
            avg_loss = total_loss / len(train_samples)
            logging.info(f"📘 SVD++ Epoch {epoch+1}/{self.epochs} 完成，Loss={avg_loss:.4f}")

            min_delta = 1e-4
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logging.info(f"🛑 提前停止训练 (Early Stop at Epoch {epoch+1})，best_loss={best_loss:.4f}")
                    break

    def _collate_fn(self, batch):
        users, items, ratings, imp_lists = zip(*batch)
        flat_implicit = torch.cat([torch.tensor(lst, dtype=torch.long) for lst in imp_lists])
        offsets = torch.tensor([0] + [len(x) for x in imp_lists[:-1]]).cumsum(0)
        return (
            torch.tensor(users, dtype=torch.long),
            torch.tensor(items, dtype=torch.long),
            torch.tensor(ratings, dtype=torch.float),
            offsets,
            flat_implicit
        )

    def recommend(self, user_id, all_items, k=10):
        if user_id not in self.user_map:
            return []
        self.model.eval()
        with torch.no_grad():
            uid = self.user_map[user_id]
            user_tensor = torch.tensor([uid] * len(all_items), device=self.device)
            item_indices = [self.item_map[i] for i in all_items if i in self.item_map]
            item_tensor = torch.tensor(item_indices, device=self.device)
            implicit = [self.user_hist[uid]] * len(item_indices)

            flat = torch.cat([torch.tensor(lst, dtype=torch.long) for lst in implicit])
            offsets = torch.tensor([0] + [len(x) for x in implicit[:-1]]).cumsum(0)
            scores = self.model(user_tensor, item_tensor, offsets.to(self.device), flat.to(self.device)).cpu().numpy()
            top_k_idx = torch.topk(torch.tensor(scores), k).indices.numpy()
            return [self.inv_item_map[item_indices[i]] for i in top_k_idx]