
import torch
import torch.nn as nn
from collections import defaultdict

class SVDEmbedding(nn.Module):
    def __init__(self, num_users, num_items, emb_size=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        dot = (u * i).sum(1, keepdim=True)
        return dot + self.user_bias(user_ids) + self.item_bias(item_ids)

class SVDRecommender:
    def __init__(self, num_users, num_items, emb_size=64, lr=0.01, epochs=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SVDEmbedding(num_users, num_items, emb_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.loss_fn = nn.MSELoss()

    def fit(self, train_data):
        self.model.train()
        user_map = {u: i for i, u in enumerate(set(u for u, _ in train_data))}
        item_map = {i: j for j, i in enumerate(set(i for _, i in train_data))}
        self.user_map = user_map
        self.item_map = item_map
        self.inv_user_map = {i: u for u, i in user_map.items()}
        self.inv_item_map = {j: i for i, j in item_map.items()}

        interactions = [(user_map[u], item_map[i]) for u, i in train_data if u in user_map and i in item_map]
        for _ in range(self.epochs):
            for u, i in interactions:
                user = torch.tensor([u], device=self.device)
                item = torch.tensor([i], device=self.device)
                rating = torch.tensor([[1.0]], device=self.device)
                pred = self.model(user, item)
                loss = self.loss_fn(pred, rating)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def recommend(self, user_id, all_items, k=10):
        if user_id not in self.user_map:
            return []
        self.model.eval()
        with torch.no_grad():
            item_indices = [self.item_map[i] for i in all_items if i in self.item_map]
            user_idx = torch.tensor([self.user_map[user_id]] * len(item_indices), device=self.device)
            items_tensor = torch.tensor(item_indices, device=self.device)
            scores = self.model(user_idx, items_tensor).squeeze().cpu().numpy()
            scored_items = list(zip(item_indices, scores))
            scored_items.sort(key=lambda x: x[1], reverse=True)
            return [self.inv_item_map[i] for i, _ in scored_items[:k]]
