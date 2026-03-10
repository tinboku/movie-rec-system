import numpy as np
from collections import Counter


class PopularityRecommender:
    def __init__(self):
        self.popular_items = []
        self.item_scores = {}

    def fit(self, train_ratings):
        # count how many times each item got rated
        counts = Counter(train_ratings['item_id'].values)
        self.popular_items = [item for item, _ in counts.most_common()]
        self.item_scores = dict(counts)

    def recommend(self, user_id, n=10, exclude_items=None):
        if exclude_items is None:
            exclude_items = set()
        recs = []
        for item in self.popular_items:
            if item not in exclude_items:
                recs.append(item)
            if len(recs) >= n:
                break
        return recs

    def recommend_all(self, user_ids, train_user_items, n=10):
        """recommend for all users, excluding items they already interacted with"""
        result = {}
        for uid in user_ids:
            exclude = train_user_items.get(uid, set())
            result[uid] = self.recommend(uid, n=n, exclude_items=exclude)
        return result
