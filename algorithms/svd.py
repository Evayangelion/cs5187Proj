import numpy as np
from scipy.sparse.linalg import svds

class SVDRecommender:
    def __init__(self, n_factors=50):
        """
        Initialize SVD-based Recommender System.
        
        Args:
            n_factors (int): Number of latent factors.
        """
        self.n_factors = n_factors
        self.user_indices = None
        self.movie_indices = None
        self.prediction_matrix = None
        self.train_index = None
        self.train_columns = None
        self.user_means = None
        self.U = None
        self.Vt = None
        self.sigma = None
        self.train_matrix = None
    
    def fit(self, train_matrix):
        """
        Train the SVD model.
        
        Args:
            train_matrix (pd.DataFrame): Training user-item rating matrix.
        
        Returns:
            self: Fitted model.
        """
        print(f"Training SVD model (number of factors={self.n_factors})...")
        self.user_indices = {uid: i for i, uid in enumerate(train_matrix.index)}
        self.movie_indices = {mid: i for i, mid in enumerate(train_matrix.columns)}
        self.train_index = train_matrix.index
        self.train_columns = train_matrix.columns
        self.train_matrix = train_matrix.values
        ratings_array = train_matrix.values
        self.user_means = np.nanmean(ratings_array, axis=1)
        ratings_centered = ratings_array.copy()
        for i in range(ratings_array.shape[0]):
            ratings_centered[i, :] = ratings_array[i, :] - self.user_means[i]
        ratings_centered = np.nan_to_num(ratings_centered)
        U, sigma, Vt = svds(ratings_centered, k=self.n_factors)
        sigma = np.abs(sigma)
        self.sigma = np.diag(sigma)
        self.U = U
        self.Vt = Vt
        self.prediction_matrix = np.dot(np.dot(U, self.sigma), Vt) + self.user_means.reshape(-1, 1)
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
        return self.prediction_matrix[user_idx, item_idx]
    
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
            return self._cold_start_recommendation(n)
        user_idx = self.user_indices[user_id]
        user_predictions = self.prediction_matrix[user_idx]
        rated_items = set()
        for item_id, item_idx in self.movie_indices.items():
            if item_idx < self.prediction_matrix.shape[1] and self.train_matrix[user_idx, item_idx] > 0:
                rated_items.add(item_id)
        predictions = []
        for item_id, item_idx in self.movie_indices.items():
            if item_id not in rated_items and item_idx < user_predictions.shape[0]:
                predicted_rating = user_predictions[item_idx]
                predictions.append((item_id, predicted_rating))
        return sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    
    def _cold_start_recommendation(self, n=10):
        """
        Handle cold start by recommending globally popular movies.
        
        Args:
            n (int): Number of recommendations.
        
        Returns:
            list: List of (item_id, score) tuples.
        """
        movie_avg_ratings = np.mean(self.prediction_matrix, axis=0)
        movie_ratings = [(movie_id, movie_avg_ratings[idx]) 
                        for movie_id, idx in self.movie_indices.items()
                        if idx < len(movie_avg_ratings)]
        return sorted(movie_ratings, key=lambda x: x[1], reverse=True)[:n]