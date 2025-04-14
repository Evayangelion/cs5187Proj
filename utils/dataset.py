import random
from collections import defaultdict

def load_movielens(path='data/ml-1m/ratings.dat', split='temporal', seed=42):
    """
    加载 MovieLens-1M 数据并进行划分
    返回: train, test, all_items, all_users
    - train: List[(user, item, rating)]
    - test: List[(user, item)]
    - all_items, all_users: 全部 item/user id
    """
    data = []
    user_set, item_set = set(), set()

    with open(path, 'r') as f:
        for line in f:
            user, item, rating, ts = line.strip().split('::')
            user, item, rating, ts = int(user), int(item), float(rating), int(ts)
            data.append((user, item, rating, ts))
            user_set.add(user)
            item_set.add(item)

    if split == 'temporal':
        data.sort(key=lambda x: x[3])  # 按时间升序
        cutoff = int(len(data) * 0.8)
        train_raw = data[:cutoff]
        test_raw = data[cutoff:]

    elif split == 'random':
        random.seed(seed)
        random.shuffle(data)
        cutoff = int(len(data) * 0.8)
        train_raw = data[:cutoff]
        test_raw = data[cutoff:]

    elif split == 'leave-one-out':
        user_hist = defaultdict(list)
        for u, i, r, t in data:
            user_hist[u].append((u, i, r, t))
        train_raw, test_raw = [], []
        for u, items in user_hist.items():
            items.sort(key=lambda x: x[3])
            if len(items) < 2:
                train_raw.extend(items)
            else:
                train_raw.extend(items[:-1])
                test_raw.append(items[-1])
    else:
        raise ValueError("split must be one of: 'temporal', 'random', 'leave-one-out'")

    # 统一格式输出
    train = [(u, i, r) for u, i, r, _ in train_raw]
    test = [(u, i) for u, i, r, _ in test_raw]
    all_users = sorted(user_set)
    all_items = sorted(item_set)

    return train, test, all_items, all_users