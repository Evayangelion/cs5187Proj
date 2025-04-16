import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from collections import defaultdict
import random

class LightGCNDataset:
    def __init__(self, data, threshold=4.0):
        """
        Process dataset for LightGCN.
        
        Args:
            data (pd.DataFrame): Raw dataset with userId, movieId, rating columns.
            threshold (float): Rating threshold for positive samples.
        """
        self.data = data
        self.threshold = threshold
        self.user_mapping = {id: i for i, id in enumerate(data['userId'].unique())}
        self.movie_mapping = {id: i for i, id in enumerate(data['movieId'].unique())}
        self.n_users = len(self.user_mapping)
        self.n_items = len(self.movie_mapping)
        self.data['user_idx'] = self.data['userId'].map(self.user_mapping)
        self.data['item_idx'] = self.data['movieId'].map(self.movie_mapping)
        self.reverse_user_mapping = {v: k for k, v in self.user_mapping.items()}
        self.reverse_movie_mapping = {v: k for k, v in self.movie_mapping.items()}
        self.train_data, self.test_dict = self._split_data()
        self.adj_tensor = self._build_graph()
    
    def _split_data(self):
        """Split training and test sets."""
        print("Splitting training and test sets...")
        train_data = []
        test_data = []
        self.data['label'] = (self.data['rating'] >= self.threshold).astype(int)
        grouped = self.data.groupby('user_idx')
        for user_idx, user_data in grouped:
            pos_items = user_data[user_data['label'] == 1]
            if len(pos_items) > 1:
                test_item = pos_items.sample(1, random_state=42)
                train_user_data = user_data[~user_data.index.isin(test_item.index)]
                train_data.append(train_user_data)
                test_data.append(test_item)
        if not train_data or not test_data:
            raise ValueError("Dataset splitting failed, possibly due to insufficient positive samples")
        train_data = pd.concat(train_data)
        test_data = pd.concat(test_data)
        test_dict = defaultdict(list)
        for _, row in test_data.iterrows():
            test_dict[row['user_idx']].append(row['item_idx'])
        return train_data, test_dict
    
    def _build_graph(self):
        """Build user-item interaction graph."""
        print("Building user-item interaction graph...")
        train_pos_data = self.train_data[self.train_data['label'] == 1]
        user_indices = train_pos_data['user_idx'].values
        item_indices = train_pos_data['item_idx'].values
        ratings = np.ones(len(user_indices), dtype=np.float32)
        R = sp.coo_matrix(
            (ratings, (user_indices, item_indices)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32
        )
        zero_uu = sp.csr_matrix((self.n_users, self.n_users), dtype=np.float32)
        zero_ii = sp.csr_matrix((self.n_items, self.n_items), dtype=np.float32)
        upper_half = sp.hstack([zero_uu, R])
        lower_half = sp.hstack([R.T, zero_ii])
        adj = sp.vstack([upper_half, lower_half])
        adj = self._normalize_adj(adj + sp.eye(adj.shape[0]))
        adj = adj.tocoo().astype(np.float32)
        indices = torch.LongTensor([adj.row, adj.col])
        values = torch.FloatTensor(adj.data)
        shape = torch.Size(adj.shape)
        return torch.sparse_coo_tensor(indices, values, shape)
    
    def _normalize_adj(self, adj):
        """Normalize adjacency matrix: D^(-1/2) A D^(-1/2)."""
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    
    def get_train_instances(self, num_negatives=4):
        """Generate training instances with negative sampling."""
        pos_items = self.train_data[self.train_data['label'] == 1]
        user_pos_items = pos_items.groupby('user_idx')['item_idx'].apply(list).to_dict()
        all_items = set(range(self.n_items))
        train_instances = []
        for user, pos_list in user_pos_items.items():
            for pos_item in pos_list:
                train_instances.append((user, pos_item, 1.0))
                for _ in range(num_negatives):
                    user_neg_items = all_items - set(user_pos_items.get(user, []))
                    if not user_neg_items:
                        continue
                    neg_item = random.sample(list(user_neg_items), 1)[0]
                    train_instances.append((user, neg_item, 0.0))
        return train_instances

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, adj_tensor, embedding_dim=64, n_layers=3):
        """
        Initialize LightGCN model.
        
        Args:
            n_users (int): Number of users.
            n_items (int): Number of items.
            adj_tensor (torch.Tensor): Adjacency matrix as sparse tensor.
            embedding_dim (int): Embedding dimension.
            n_layers (int): Number of graph convolution layers.
        """
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        self.adj = adj_tensor
    
    def forward(self):
        """Forward propagation."""
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        embeddings_list = [all_embeddings]
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        all_embeddings = torch.stack(embeddings_list, dim=0).mean(dim=0)
        user_all_embeddings, item_all_embeddings = torch.split(
            all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings
    
    def calculate_loss(self, users, items, labels):
        """Calculate BPR loss."""
        user_embeddings, item_embeddings = self.forward()
        user_emb = user_embeddings[users]
        item_emb = item_embeddings[items]
        scores = torch.sum(user_emb * item_emb, dim=1)
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        l2_reg = 1e-5 * (user_emb.norm(2).pow(2) + item_emb.norm(2).pow(2))
        return loss + l2_reg
    
    def predict(self, users, items=None):
        """Predict user ratings for items."""
        user_embeddings, item_embeddings = self.forward()
        user_emb = user_embeddings[users]
        if items is not None:
            item_emb = item_embeddings[items]
            return torch.sum(user_emb * item_emb, dim=1)
        return torch.matmul(user_emb, item_embeddings.T)

class LightGCNRecommender:
    def __init__(self, embedding_dim=64, n_layers=3, batch_size=1024, lr=0.001, epochs=10, num_negatives=4, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize LightGCN Recommender.
        
        Args:
            embedding_dim (int): Embedding dimension.
            n_layers (int): Number of graph convolution layers.
            batch_size (int): Batch size for training.
            lr (float): Learning rate.
            epochs (int): Number of training epochs.
            num_negatives (int): Number of negative samples per positive sample.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.num_negatives = num_negatives
        self.device = device
        self.model = None
        self.dataset = None
        self.user_indices = None
        self.movie_indices = None
    
    def fit(self, train_data):
        """
        Train the LightGCN model.
        
        Args:
            train_data (pd.DataFrame): Training data with userId, movieId, rating columns.
        
        Returns:
            self: Fitted model.
        """
        print("Initializing LightGCN dataset...")
        self.dataset = LightGCNDataset(train_data)
        self.user_indices = self.dataset.user_mapping
        self.movie_indices = self.dataset.movie_mapping
        self.model = LightGCN(
            self.dataset.n_users,
            self.dataset.n_items,
            self.dataset.adj_tensor.to(self.device),
            self.embedding_dim,
            self.n_layers
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        train_instances = self.dataset.get_train_instances(self.num_negatives)
        print(f"Starting training (Device: {self.device})...")
        best_hr = 0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            random.shuffle(train_instances)
            total_loss = 0
            n_batches = len(train_instances) // self.batch_size + 1
            for i in range(0, len(train_instances), self.batch_size):
                batch = train_instances[i:i+self.batch_size]
                if not batch:
                    continue
                users, items, labels = zip(*batch)
                users = torch.LongTensor(users).to(self.device)
                items = torch.LongTensor(items).to(self.device)
                labels = torch.FloatTensor(labels).to(self.device)
                optimizer.zero_grad()
                loss = self.model.calculate_loss(users, items, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch}/{self.epochs} - Loss: {total_loss / n_batches:.4f}")
        return self
    
    def predict(self, user_id, item_id):
        """
        Predict user's rating for an item.
        
        Args:
            user_id: User ID.
            item_id: Item (movie) ID.
        
        Returns:
            float: Predicted rating.
        """
        if user_id not in self.user_indices or item_id not in self.movie_indices:
            return 0
        user_idx = self.user_indices[user_id]
        item_idx = self.movie_indices[item_id]
        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx]).to(self.device)
            item_tensor = torch.LongTensor([item_idx]).to(self.device)
            score = self.model.predict(user_tensor, item_tensor)
        return score.item()
    
    def recommend_for_all_users(self, test_dict, n=10):
        """
        Generate recommendations for all test users.
        
        Args:
            test_dict (dict): Dictionary of user-items in test set {user_id: [item_ids]}.
            n (int): Number of recommendations per user.
        
        Returns:
            dict: Dictionary of recommendations {user_id: [(item_id, score)]}.
        """
        print("Generating recommendations for all users...")
        reco_dict = {}
        from tqdm import tqdm
        self.model.eval()
        with torch.no_grad():
            for user_id in tqdm(test_dict.keys()):
                recommendations = self.recommend_for_user(user_id, n)
                if recommendations:
                    reco_dict[user_id] = recommendations
        return reco_dict
    
    def recommend_for_user(self, user_id, n=10):
        """
        Generate Top-N recommendations for a single user.
        
        Args:
            user_id: User ID.
            n (int): Number of recommendations.
        
        Returns:
            list: List of (item_id, score) tuples.
        """
        if user_id not in self.user_indices:
            return []
        user_idx = self.user_indices[user_id]
        train_items = set(self.dataset.train_data[
            (self.dataset.train_data['user_idx'] == user_idx) & 
            (self.dataset.train_data['label'] == 1)
        ]['item_idx'].tolist())
        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx]).to(self.device)
            predictions = self.model.predict(user_tensor)[0].cpu().numpy()
            for item in train_items:
                predictions[item] = -float('inf')
            topn_indices = np.argsort(-predictions)[:n]
            recs = []
            for idx in topn_indices:
                movie_id = self.dataset.reverse_movie_mapping[idx]
                score = float(predictions[idx])
                recs.append((movie_id, score))
        return recs