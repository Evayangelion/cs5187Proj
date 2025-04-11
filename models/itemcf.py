import numpy as np
from collections import defaultdict
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class ItemCF:
    def __init__(self, k=20):
        self.k = k

    def fit(self, train_data):
        user_item = defaultdict(set)
        for user, item in train_data:
            user_item[user].add(item)

        all_items = set(i for _, i in train_data)
        all_users = list(user_item.keys())
        item_index = {item: idx for idx, item in enumerate(all_items)}
        user_index = {user: idx for idx, user in enumerate(all_users)}

        matrix = lil_matrix((len(item_index), len(user_index)), dtype=np.float32)
        for user in user_item:
            u_idx = user_index[user]
            for item in user_item[user]:
                i_idx = item_index[item]
                matrix[i_idx, u_idx] = 1.0

        self.item_index = item_index
        self.index_item = {v: k for k, v in item_index.items()}
        self.user_item = user_item

        matrix = csr_matrix(matrix)
        self.sim_matrix = cosine_similarity(matrix, dense_output=False)

    def recommend(self, user, k=10):
        interacted = self.user_item.get(user, set())
        scores = defaultdict(float)
        for item in interacted:
            idx = self.item_index.get(item, -1)
            if idx == -1:
                continue
            sim_items = self.sim_matrix.getrow(idx).tocoo()
            for i, score in zip(sim_items.col, sim_items.data):
                candidate = self.index_item[i]
                if candidate not in interacted:
                    scores[candidate] += score
        return sorted(scores, key=scores.get, reverse=True)[:k]
