import pandas as pd
from collections import defaultdict

def preprocess_data(data, threshold=4.0):
    """
    Preprocess ratings data.
    
    Args:
        data (pd.DataFrame): Raw ratings data.
        threshold (float): Rating threshold for positive samples.
    
    Returns:
        pd.DataFrame: Rating matrix.
    """
    data['like'] = (data['rating'] >= threshold).astype(int)
    rating_matrix = data.pivot_table(
        index='userId',
        columns='movieId',
        values='rating',
        fill_value=0
    )
    return rating_matrix

def split_data(data, test_size=1):
    """
    Split data into train and test sets.
    
    Args:
        data (pd.DataFrame): Ratings data with 'like' column.
        test_size (int): Number of positive samples per user for test set.
    
    Returns:
        tuple: (train_matrix, test_dict)
    """
    test_ratings = []
    train_ratings = []
    grouped = data.groupby('userId')
    
    for user_id, user_data in grouped:
        liked_items = user_data[user_data['like'] == 1]
        if len(liked_items) > test_size:
            test_item = liked_items.sample(test_size)
            train_item = user_data[~user_data.index.isin(test_item.index)]
            test_ratings.append(test_item)
            train_ratings.append(train_item)
    
    if not test_ratings or not train_ratings:
        raise ValueError("Dataset splitting failed: insufficient positive samples")
    
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
    
    return train_matrix, test_dict