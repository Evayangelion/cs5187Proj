import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

class ItemCF:
    def __init__(self, k=50):
        """
        Initialize Item-based Collaborative Filtering.
        
        Args:
            k (int): Number of neighbors.
        """
        self.k = k
        self.item_sim = None
        self.train_matrix = None
        self.user_indices = None
        self.movie_indices = None
        self.movie_id_list = None
    
    def fit(self, train_matrix):
        """
        Calculate item similarity matrix.
        
        Args:
            train_matrix (pd.DataFrame): Training user-item rating matrix.
        
        Returns:
            self: Fitted model.
        """
        print("Calculating item similarities...")
        self.user_indices = {uid: i for i, uid in enumerate(train_matrix.index)}
        self.movie_indices = {mid: i for i, mid in enumerate(train_matrix.columns)}
        self.movie_id_list = list(train_matrix.columns)
        self.item_sim = cosine_similarity(train_matrix.T)
        self.train_matrix = train_matrix.values
        self.train_index = train_matrix.index
        self.train_columns = train_matrix.columns
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
        sim_items = np.argsort(-self.item_sim[item_idx])[1:self.k+1]
        user_ratings = self.train_matrix[user_idx, sim_items]
        similarities = self.item_sim[item_idx, sim_items]
        numerator = np.sum(similarities * user_ratings)
        denominator = np.sum(np.abs(similarities)) + 1e-8
        return numerator / denominator
    
    def recommend_for_all_users(self, test_dict, n=10, method='standard'):
        """
        Generate recommendations for all test users.
        
        Args:
            test_dict (dict): Dictionary of user-items in test set {user_id: [item_ids]}.
            n (int): Number of recommendations per user.
            method (str): 'standard' or 'similarity' for recommendation algorithm.
        
        Returns:
            dict: Dictionary of recommendations {user_id: [(item_id, score)]}.
        """
        print(f"Generating {method} recommendations for all users...")
        reco_dict = {}
        from tqdm import tqdm
        for user_id in tqdm(test_dict.keys()):
            if user_id not in self.user_indices:
                continue
            if method == 'standard':
                recommendations = self.recommend_for_user(user_id, n)
            else:
                recommendations = self.similarity_based_recommend(user_id, n)
            if recommendations:
                reco_dict[user_id] = recommendations
        return reco_dict
    
    def recommend_for_user(self, user_id, n=10):
        """
        Generate Top-N recommendations for a single user using standard method.
        
        Args:
            user_id: User ID.
            n (int): Number of recommendations.
        
        Returns:
            list: List of (item_id, score) tuples.
        """
        if user_id not in self.user_indices:
            return []
        user_idx = self.user_indices[user_id]
        rated_items = set()
        for item_id, item_idx in self.movie_indices.items():
            if self.train_matrix[user_idx, item_idx] > 0:
                rated_items.add(item_id)
        predictions = []
        for item_id in self.movie_indices.keys():
            if item_id not in rated_items:
                predicted_rating = self.predict(user_id, item_id)
                predictions.append((item_id, predicted_rating))
        return sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    
    def similarity_based_recommend(self, user_id, n=10):
        """
        Generate Top-N recommendations using similarity-based method.
        
        Args:
            user_id: User ID.
            n (int): Number of recommendations.
        
        Returns:
            list: List of (item_id, score) tuples.
        """
        if user_id not in self.user_indices:
            return []
        user_idx = self.user_indices[user_id]
        rated_items = []
        rated_indices = []
        for item_id, item_idx in self.movie_indices.items():
            rating = self.train_matrix[user_idx, item_idx]
            if rating > 0:
                rated_items.append((item_id, rating, item_idx))
        if not rated_items:
            return []
        unrated_items = set(self.movie_indices.keys()) - set(item[0] for item in rated_items)
        candidate_scores = defaultdict(float)
        for item_id, rating, item_idx in rated_items:
            for other_idx, sim_score in enumerate(self.item_sim[item_idx]):
                other_item_id = self.movie_id_list[other_idx]
                if other_item_id in unrated_items:
                    candidate_scores[other_item_id] += sim_score * rating
        recommendations = [(item, score) for item, score in candidate_scores.items()]
        return sorted(recommendations, key=lambda x: x[1], reverse=True)[:n]