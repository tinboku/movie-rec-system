import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import os
import yaml


GENRE_COLS = [
    'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]


def load_config(path='configs/config.yaml'):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_ratings(data_dir='data/raw'):
    fpath = os.path.join(data_dir, 'u.data')
    df = pd.read_csv(fpath, sep='\t', header=None,
                     names=['user_id', 'item_id', 'rating', 'timestamp'])
    return df


def load_items(data_dir='data/raw'):
    fpath = os.path.join(data_dir, 'u.item')
    cols = ['item_id', 'title', 'release_date', 'video_release', 'url'] + GENRE_COLS
    df = pd.read_csv(fpath, sep='|', header=None, names=cols,
                     encoding='latin-1')
    return df


def load_users(data_dir='data/raw'):
    fpath = os.path.join(data_dir, 'u.user')
    df = pd.read_csv(fpath, sep='|', header=None,
                     names=['user_id', 'age', 'gender', 'occupation', 'zip'])
    return df


def train_test_split_by_time(ratings, test_ratio=0.2):
    """split by timestamp, last 20% of each user's ratings go to test"""
    ratings = ratings.sort_values(['user_id', 'timestamp'])
    train_list, test_list = [], []
    for uid, group in ratings.groupby('user_id'):
        n = len(group)
        n_test = max(1, int(n * test_ratio))
        train_list.append(group.iloc[:-n_test])
        test_list.append(group.iloc[-n_test:])
    train = pd.concat(train_list).reset_index(drop=True)
    test = pd.concat(test_list).reset_index(drop=True)
    return train, test


def train_test_split_random(ratings, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    train_list, test_list = [], []
    for uid, group in ratings.groupby('user_id'):
        n = len(group)
        n_test = max(1, int(n * test_ratio))
        idx = np.random.permutation(n)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        train_list.append(group.iloc[train_idx])
        test_list.append(group.iloc[test_idx])
    return (pd.concat(train_list).reset_index(drop=True),
            pd.concat(test_list).reset_index(drop=True))


def build_interaction_matrix(ratings, n_users=943, n_items=1682):
    """build sparse user-item matrix"""
    row = ratings['user_id'].values - 1  # 0-indexed
    col = ratings['item_id'].values - 1
    vals = ratings['rating'].values.astype(np.float32)
    mat = csr_matrix((vals, (row, col)), shape=(n_users, n_items))
    return mat


def build_implicit_matrix(ratings, n_users=943, n_items=1682, threshold=4):
    row = ratings['user_id'].values - 1
    col = ratings['item_id'].values - 1
    vals = (ratings['rating'].values >= threshold).astype(np.float32)
    mat = csr_matrix((vals, (row, col)), shape=(n_users, n_items))
    return mat


def get_user_positive_items(ratings, threshold=4):
    """returns dict: user_id -> set of item_ids that user liked"""
    pos = ratings[ratings['rating'] >= threshold]
    result = {}
    for uid, group in pos.groupby('user_id'):
        result[uid] = set(group['item_id'].values)
    return result
