import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.metrics import hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k, evaluate_rankings


def test_hit_rate_basic():
    ranked = [1, 2, 3, 4, 5]
    gt = [3, 7]
    assert hit_rate_at_k(ranked, gt, k=5) == 1.0
    assert hit_rate_at_k(ranked, gt, k=2) == 0.0


def test_ndcg_perfect():
    ranked = [1, 2, 3]
    gt = [1, 2, 3]
    score = ndcg_at_k(ranked, gt, k=3)
    assert abs(score - 1.0) < 1e-6


def test_ndcg_partial():
    ranked = [5, 1, 6, 2, 7]
    gt = [1, 2]
    score = ndcg_at_k(ranked, gt, k=5)
    assert 0 < score < 1.0


def test_precision():
    ranked = [1, 2, 3, 4, 5]
    gt = [1, 3, 5]
    assert abs(precision_at_k(ranked, gt, k=5) - 0.6) < 1e-6
    assert abs(precision_at_k(ranked, gt, k=2) - 0.5) < 1e-6


def test_recall():
    ranked = [1, 2, 3, 4, 5]
    gt = [1, 3, 8]
    assert abs(recall_at_k(ranked, gt, k=5) - 2/3) < 1e-6


def test_recall_empty_gt():
    ranked = [1, 2, 3]
    assert recall_at_k(ranked, [], k=3) == 0.0


def test_evaluate_rankings():
    rankings = {
        1: [10, 20, 30, 40, 50],
        2: [1, 2, 3, 4, 5],
    }
    gt = {
        1: [10, 20],
        2: [99],
    }
    result = evaluate_rankings(rankings, gt, k=5)
    assert result['HR@5'] == 0.5  # user1 hit, user2 miss
    assert result['Precision@5'] > 0
