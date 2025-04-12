from collections import Counter

class PopularityRecommender:
    def fit(self, train_data):
        from collections import defaultdict
        counts = defaultdict(int)
        for _, item, _ in train_data:
            counts[item] += 1
        self.top_items = sorted(counts, key=counts.get, reverse=True)

    def recommend(self, user, k=10):
        return self.top_items[:k]