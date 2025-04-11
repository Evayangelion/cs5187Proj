import pandas as pd

def load_movielens(path='data/ml-1m/ratings.dat'):
    df = pd.read_csv(path, sep='::', engine='python',
                     names=['userId', 'movieId', 'rating', 'timestamp'])
    df = df.sort_values(by=['userId', 'timestamp'])
    
    train, test = [], []
    for user, group in df.groupby('userId'):
        items = group['movieId'].tolist()
        if len(items) < 2:
            continue
        test.append((user, items[-1]))
        for item in items[:-1]:
            train.append((user, item))
    return train, test, df['movieId'].unique(), df['userId'].unique()
