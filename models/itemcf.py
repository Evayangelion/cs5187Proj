import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

class ItemCF:
    def __init__(self, k=20):
        self.k = k

    def fit(self, train_data):
        user_item = defaultdict(set)
        for user, item in train_data:
            user_item[user].add(item)
        items = set(i for _, i in train_data)
        item_index = {item: idx for idx, item in enumerate(items)}

        matrix = np.zeros((len(items), len(user_item)))
        for col, user in enumerate(user_item):
            for item in user_item[user]:
                matrix[item_index[item]][col] = 1

        self.item_index = item_index
        self.index_item = {v: k for k, v in item_index.items()}
        self.sim_matrix = cosine_similarity(matrix)
        self.user_item = user_item

    def recommend(self, user, k=10):
        interacted = self.user_item.get(user, set())
        scores = defaultdict(float)
        for item in interacted:
            idx = self.item_index.get(item, -1)
            if idx == -1: continue
            for i, score in enumerate(self.sim_matrix[idx]):
                candidate = self.index_item[i]
                if candidate not in interacted:
                    scores[candidate] += score
        return sorted(scores, key=scores.get, reverse=True)[:k]