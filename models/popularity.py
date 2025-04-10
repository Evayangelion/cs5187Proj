from collections import Counter

class PopularityRecommender:
    def fit(self, train_data):
        items = [item for _, item in train_data]
        self.top_items = [item for item, _ in Counter(items).most_common()]

    def recommend(self, user, k=10):
        return self.top_items[:k]
