import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


class UserBasedCF:
    def __init__(self, k=20):
        self.k = k
        self.user_sim = None
        self.interaction_mat = None

    def fit(self, interaction_matrix):
        self.interaction_mat = interaction_matrix
        # compute user-user similarity
        self.user_sim = cosine_similarity(interaction_matrix)
        np.fill_diagonal(self.user_sim, 0)

    def predict_scores(self, user_idx):
        # weighted sum of similar users' ratings
        sim = self.user_sim[user_idx]
        # get top-k similar users
        topk_users = np.argsort(sim)[-self.k:]
        topk_sims = sim[topk_users]

        mat = self.interaction_mat
        if isinstance(mat, csr_matrix):
            mat = mat.toarray()

        scores = np.zeros(mat.shape[1])
        sim_sum = np.sum(np.abs(topk_sims))
        if sim_sum > 0:
            for i, u in enumerate(topk_users):
                scores += topk_sims[i] * mat[u]
            scores /= (sim_sum + 1e-8)
        return scores

    def recommend(self, user_idx, n=10, exclude_items=None):
        scores = self.predict_scores(user_idx)
        if exclude_items:
            for item_idx in exclude_items:
                scores[item_idx] = -np.inf
        topn = np.argsort(scores)[::-1][:n]
        return topn.tolist()

    def recommend_all_users(self, n_users, train_user_items_idx, n=10):
        result = {}
        for uid_idx in range(n_users):
            exclude = train_user_items_idx.get(uid_idx, set())
            recs = self.recommend(uid_idx, n=n, exclude_items=exclude)
            # convert back to 1-indexed
            result[uid_idx + 1] = [r + 1 for r in recs]
        return result


class ItemBasedCF:
    def __init__(self, k=20):
        self.k = k
        self.item_sim = None
        self.interaction_mat = None

    def fit(self, interaction_matrix):
        self.interaction_mat = interaction_matrix
        # item-item similarity (transpose the matrix)
        self.item_sim = cosine_similarity(interaction_matrix.T)
        np.fill_diagonal(self.item_sim, 0)

    def predict_scores(self, user_idx):
        mat = self.interaction_mat
        if isinstance(mat, csr_matrix):
            mat = mat.toarray()
        user_ratings = mat[user_idx]  # shape: (n_items,)

        rated_items = np.where(user_ratings > 0)[0]
        scores = np.zeros(mat.shape[1])

        for item_idx in range(mat.shape[1]):
            if user_ratings[item_idx] > 0:
                continue
            # find top-k similar items among rated items
            sims = self.item_sim[item_idx][rated_items]
            if len(sims) == 0:
                continue
            topk_idx = np.argsort(sims)[-self.k:]
            topk_sims = sims[topk_idx]
            topk_ratings = user_ratings[rated_items[topk_idx]]
            denom = np.sum(np.abs(topk_sims))
            if denom > 0:
                scores[item_idx] = np.dot(topk_sims, topk_ratings) / denom
        return scores

    def recommend(self, user_idx, n=10, exclude_items=None):
        scores = self.predict_scores(user_idx)
        if exclude_items:
            for item_idx in exclude_items:
                scores[item_idx] = -np.inf
        topn = np.argsort(scores)[::-1][:n]
        return topn.tolist()

    def recommend_all_users(self, n_users, train_user_items_idx, n=10):
        result = {}
        for uid_idx in range(n_users):
            exclude = train_user_items_idx.get(uid_idx, set())
            recs = self.recommend(uid_idx, n=n, exclude_items=exclude)
            result[uid_idx + 1] = [r + 1 for r in recs]
        return result
