import pandas as pd
from collections import defaultdict

class RecDataset:
    def __init__(self, data_path, dataset_name='movielens', threshold=4.0):
        """
        Load and preprocess dataset (MovieLens, LastFM, or Yelp).
        
        Args:
            data_path (str): Path to dataset file (e.g., ratings.csv for MovieLens).
            dataset_name (str): Name of dataset.
            threshold (float): Rating threshold for positive samples.
        """
        print("Loading data...")
        self.dataset_name = dataset_name
        self.threshold = threshold
        self.data = self.load_data(data_path)
        self.train_matrix, self.test_dict, self.train_data = self.preprocess_data()

    def load_data(self, data_path):
        """Load dataset."""
        if self.dataset_name == 'movielens':
            ratings = pd.read_csv(data_path)
            ratings = ratings[['userId', 'movieId', 'rating']]
            return ratings
        raise NotImplementedError(f"Dataset {self.dataset_name} not supported yet")

    def preprocess_data(self):
        """Split data into training and test sets and create user-item matrix."""
        print("Splitting training and test sets...")
        test_ratings = []
        train_ratings = []
        ratings = self.data.copy()
        ratings['like'] = (ratings['rating'] >= self.threshold).astype(int)
        grouped = ratings.groupby('userId')
        for user_id, user_data in grouped:
            liked_items = user_data[user_data['like'] == 1]
            if len(liked_items) > 1:
                test_item = liked_items.sample(1, random_state=42)
                train_item = user_data[~user_data.index.isin(test_item.index)]
                test_ratings.append(test_item)
                train_ratings.append(train_item)
        if not test_ratings or not train_ratings:
            raise ValueError("Dataset splitting failed, possibly due to insufficient positive samples")
        test = pd.concat(test_ratings)
        train = pd.concat(train_ratings)
        train_matrix = train.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=0
        )
        test_dict = defaultdict(list)
        for _, row in test.iterrows():
            test_dict[row['userId']].append(row['movieId'])
        return train_matrix, test_dict, train