import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix


class SVDRecommender:
    """Matrix factorization via truncated SVD"""

    def __init__(self, n_factors=50):
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.global_mean = 0
        self.pred_matrix = None

    def fit(self, interaction_matrix):
        mat = interaction_matrix
        if isinstance(mat, csr_matrix):
            dense = mat.toarray().astype(np.float64)
        else:
            dense = mat.astype(np.float64)

        self.global_mean = dense[dense > 0].mean()

        # center the matrix (only non-zero entries)
        centered = dense.copy()
        centered[dense > 0] -= self.global_mean

        k = min(self.n_factors, min(dense.shape) - 1)
        U, sigma, Vt = svds(csr_matrix(centered), k=k)

        # sort by singular values descending
        idx = np.argsort(-sigma)
        U = U[:, idx]
        sigma = sigma[idx]
        Vt = Vt[idx, :]

        # filter out near-zero singular values
        mask = sigma > 1e-9
        U = U[:, mask]
        sigma = sigma[mask]
        Vt = Vt[mask, :]

        self.user_factors = U
        self.sigma = sigma
        self.item_factors = Vt.T  # (n_items, k)

        # reconstruct
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            self.pred_matrix = self.global_mean + U @ np.diag(sigma) @ Vt
        self.pred_matrix = np.nan_to_num(self.pred_matrix, nan=self.global_mean)
        self.pred_matrix = np.clip(self.pred_matrix, 1, 5)

    def predict(self, user_idx, item_idx):
        return self.pred_matrix[user_idx, item_idx]

    def recommend(self, user_idx, n=10, exclude_items=None):
        scores = self.pred_matrix[user_idx].copy()
        if exclude_items:
            for idx in exclude_items:
                scores[idx] = -np.inf
        topn = np.argsort(scores)[::-1][:n]
        return topn.tolist()

    def recommend_all_users(self, n_users, train_user_items_idx, n=10):
        result = {}
        for uid_idx in range(n_users):
            exclude = train_user_items_idx.get(uid_idx, set())
            recs = self.recommend(uid_idx, n=n, exclude_items=exclude)
            result[uid_idx + 1] = [r + 1 for r in recs]
        return result
