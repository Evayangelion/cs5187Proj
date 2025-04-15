def load_lastfm(path='data/lastfm/user_artists.dat', split='leave-one-out'):
    data = []
    user_set, item_set = set(), set()

    with open(path, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            user, item, weight = map(int, line.strip().split('\t'))
            data.append((user, item, float(weight), 0))  # dummy timestamp
            user_set.add(user)
            item_set.add(item)

    if split == 'leave-one-out':
        from collections import defaultdict
        user_hist = defaultdict(list)
        for u, i, r, t in data:
            user_hist[u].append((u, i, r, t))
        train_raw, test_raw = [], []
        for u, items in user_hist.items():
            items.sort(key=lambda x: x[2])  # by weight
            if len(items) < 2:
                train_raw.extend(items)
            else:
                train_raw.extend(items[:-1])
                test_raw.append(items[-1])

    train = [(u, i, r) for u, i, r, _ in train_raw]
    test = [(u, i) for u, i, r, _ in test_raw]
    all_users = sorted(user_set)
    all_items = sorted(item_set)
    return train, test, all_items, all_users