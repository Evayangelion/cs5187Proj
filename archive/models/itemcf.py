# ================ models/itemcf.py ====================
import numpy as np
import logging
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

class ItemCF:
    def __init__(self):
        self.user_item = defaultdict(set)
        self.sim_matrix = None
        self.item_map = {}
        self.inv_item_map = {}
        self.user_matrix = None

    def fit(self, train_data):
        logging.info("构建用户-物品矩阵...")
        item_set = set()
        user_set = set()
        for u, i, _ in train_data:
            self.user_item[u].add(i)
            user_set.add(u)
            item_set.add(i)

        item_list = sorted(list(item_set))
        self.item_map = {i: idx for idx, i in enumerate(item_list)}
        self.inv_item_map = {idx: i for i, idx in self.item_map.items()}
        user_list = sorted(list(user_set))
        self.user_map = {u: idx for idx, u in enumerate(user_list)}
        self.inv_user_map = {idx: u for u, idx in self.user_map.items()}

        num_items = len(item_list)
        num_users = len(user_list)
        matrix = np.zeros((num_users, num_items))

        for u, i, _ in train_data:
            if i in self.item_map and u in self.user_map:
                matrix[self.user_map[u], self.item_map[i]] = 1.0

        logging.info("计算 item-item 相似度矩阵...")
        self.sim_matrix = cosine_similarity(matrix.T)
        self.user_matrix = matrix  # 保存用户-物品稀疏矩阵
        logging.info("相似度矩阵 shape: %s", self.sim_matrix.shape)

    def recommend(self, user, k=10):
        if user not in self.user_map:
            return []

        user_idx = self.user_map[user]
        user_vector = self.user_matrix[user_idx]  # shape: (num_items,)

        # 得分 = 用户历史评分向量 × item 相似度矩阵
        scores = np.dot(user_vector, self.sim_matrix)  # shape: (num_items,)

        # 排除已交互
        for i in self.user_item[user]:
            if i in self.item_map:
                scores[self.item_map[i]] = -np.inf

        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [self.inv_item_map[idx] for idx in top_indices if idx in self.inv_item_map]
