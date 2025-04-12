import pandas as pd

def load_movielens(path='data/ml-1m/ratings.dat'):
    df = pd.read_csv(path, sep='::', engine='python',
                     names=['userId', 'movieId', 'rating', 'timestamp'])
    df = df.sort_values(by=['userId', 'timestamp'])

    train, test = [], []
    for user, group in df.groupby('userId'):
        pairs = group[['movieId', 'rating']].values.tolist()
        if len(pairs) < 2:
            continue
        test.append((user, pairs[-1][0]))
        for item, rating in pairs[:-1]:
            train.append((user, item, rating))

    return train, test, df['movieId'].unique(), df['userId'].unique()
