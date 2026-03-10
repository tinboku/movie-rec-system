import numpy as np


def hit_rate_at_k(ranked_list, ground_truth, k=10):
    topk = set(ranked_list[:k])
    return 1.0 if len(topk & set(ground_truth)) > 0 else 0.0


def ndcg_at_k(ranked_list, ground_truth, k=10):
    topk = ranked_list[:k]
    gt_set = set(ground_truth)
    dcg = 0.0
    for i, item in enumerate(topk):
        if item in gt_set:
            dcg += 1.0 / np.log2(i + 2)
    # ideal dcg
    n_rel = min(len(gt_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_rel))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def precision_at_k(ranked_list, ground_truth, k=10):
    topk = set(ranked_list[:k])
    gt_set = set(ground_truth)
    if k == 0:
        return 0.0
    return len(topk & gt_set) / k


def recall_at_k(ranked_list, ground_truth, k=10):
    topk = set(ranked_list[:k])
    gt_set = set(ground_truth)
    if len(gt_set) == 0:
        return 0.0
    return len(topk & gt_set) / len(gt_set)


def evaluate_rankings(user_rankings, user_ground_truth, k=10):
    """
    user_rankings: dict user_id -> ranked item list
    user_ground_truth: dict user_id -> list of relevant items
    """
    hrs, ndcgs, precs, recs = [], [], [], []
    for uid in user_rankings:
        if uid not in user_ground_truth:
            continue
        gt = user_ground_truth[uid]
        if len(gt) == 0:
            continue
        ranked = user_rankings[uid]
        hrs.append(hit_rate_at_k(ranked, gt, k))
        ndcgs.append(ndcg_at_k(ranked, gt, k))
        precs.append(precision_at_k(ranked, gt, k))
        recs.append(recall_at_k(ranked, gt, k))
    return {
        'HR@{}'.format(k): np.mean(hrs) if hrs else 0,
        'NDCG@{}'.format(k): np.mean(ndcgs) if ndcgs else 0,
        'Precision@{}'.format(k): np.mean(precs) if precs else 0,
        'Recall@{}'.format(k): np.mean(recs) if recs else 0,
    }
