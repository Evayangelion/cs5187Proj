import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class UserCF:
    def __init__(self, k=50):
        """
        Initialize User-based Collaborative Filtering.
        
        Args:
            k (int): Number of neighbors.
        """
        self.k = k
        self.user_sim = None
        self.train_matrix = None
        self.user_indices = None
        self.movie_indices = None
    
    def fit(self, train_matrix):
        """
        Calculate user similarity matrix.
        
        Args:
            train_matrix (pd.DataFrame): Training user-item rating matrix.
        
        Returns:
            self: Fitted model.
        """
        print("Calculating user similarities...")
        self.user_indices = {uid: i for i, uid in enumerate(train_matrix.index)}
        self.movie_indices = {mid: i for i, mid in enumerate(train_matrix.columns)}
        self.user_sim = cosine_similarity(train_matrix)
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
        sim_users = np.argsort(-self.user_sim[user_idx])[1:self.k+1]
        numerator = np.sum(self.user_sim[user_idx][sim_users] * self.train_matrix[sim_users, item_idx])
        denominator = np.sum(np.abs(self.user_sim[user_idx][sim_users])) + 1e-8
        return numerator / denominator
    
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
        for user_id in tqdm(test_dict.keys()):
            if user_id not in self.user_indices:
                continue
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