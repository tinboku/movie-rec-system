import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.data_loader import (load_ratings, load_items, load_users,
                              train_test_split_by_time, build_interaction_matrix)


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')


def test_load_ratings():
    df = load_ratings(DATA_DIR)
    assert len(df) == 100000
    assert list(df.columns) == ['user_id', 'item_id', 'rating', 'timestamp']
    assert df['rating'].min() == 1
    assert df['rating'].max() == 5


def test_load_items():
    df = load_items(DATA_DIR)
    assert len(df) == 1682
    assert 'title' in df.columns


def test_load_users():
    df = load_users(DATA_DIR)
    assert len(df) == 943


def test_train_test_split():
    ratings = load_ratings(DATA_DIR)
    train, test = train_test_split_by_time(ratings, test_ratio=0.2)
    assert len(train) + len(test) == len(ratings)
    # every user should appear in both
    assert set(train['user_id'].unique()) == set(ratings['user_id'].unique())


def test_interaction_matrix():
    ratings = load_ratings(DATA_DIR)
    mat = build_interaction_matrix(ratings)
    assert mat.shape == (943, 1682)
    assert mat.nnz == 100000
