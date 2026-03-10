"""
Run all models and compare performance.
Usage: python run_experiment.py
"""
import sys
import os
import time
import numpy as np

from src.data_loader import (load_ratings, load_config, train_test_split_by_time,
                              train_test_split_random, build_interaction_matrix,
                              get_user_positive_items)
from src.metrics import evaluate_rankings
from src.utils import set_seed
from src.models.popularity import PopularityRecommender
from src.models.knn_cf import UserBasedCF, ItemBasedCF
from src.models.mf import SVDRecommender


def get_train_user_items_idx(train):
    """get 0-indexed user->set(item_idx) mapping for excluding seen items"""
    result = {}
    for _, row in train.iterrows():
        uid_idx = int(row['user_id']) - 1
        iid_idx = int(row['item_id']) - 1
        if uid_idx not in result:
            result[uid_idx] = set()
        result[uid_idx].add(iid_idx)
    return result


def main():
    cfg = load_config('configs/config.yaml')
    set_seed(cfg['seed'])

    print("Loading data...")
    ratings = load_ratings(cfg['data']['raw_dir'])
    print(f"  {len(ratings)} ratings, {ratings['user_id'].nunique()} users, "
          f"{ratings['item_id'].nunique()} items")

    # split
    if cfg['data']['split_method'] == 'time':
        train, test = train_test_split_by_time(ratings, cfg['data']['test_ratio'])
    else:
        train, test = train_test_split_random(ratings, cfg['data']['test_ratio'], seed=cfg['seed'])
    print(f"  train: {len(train)}, test: {len(test)}")

    k = cfg['eval']['k']
    threshold = cfg['eval']['threshold']

    # ground truth: for each user, items they rated >= threshold in test
    test_gt = {}
    for uid, group in test.groupby('user_id'):
        pos = group[group['rating'] >= threshold]['item_id'].tolist()
        if pos:
            test_gt[uid] = pos

    train_user_items = {}
    for uid, group in train.groupby('user_id'):
        train_user_items[uid] = set(group['item_id'].values)

    results = {}

    # --- Popularity ---
    print("\n[1/4] Popularity baseline...")
    t0 = time.time()
    pop = PopularityRecommender()
    pop.fit(train)
    pop_recs = pop.recommend_all(test_gt.keys(), train_user_items, n=k)
    pop_metrics = evaluate_rankings(pop_recs, test_gt, k)
    results['Popularity'] = pop_metrics
    print(f"  done in {time.time()-t0:.1f}s")
    print(f"  {pop_metrics}")

    # --- User-based CF ---
    print("\n[2/4] User-based CF...")
    t0 = time.time()
    mat = build_interaction_matrix(train, cfg['data']['n_users'], cfg['data']['n_items'])
    train_items_idx = get_train_user_items_idx(train)

    ubcf = UserBasedCF(k=cfg['models']['knn']['k'])
    ubcf.fit(mat)
    ubcf_recs = ubcf.recommend_all_users(cfg['data']['n_users'], train_items_idx, n=k)
    # filter to only users in test_gt
    ubcf_recs = {u: ubcf_recs[u] for u in test_gt if u in ubcf_recs}
    ubcf_metrics = evaluate_rankings(ubcf_recs, test_gt, k)
    results['UserCF'] = ubcf_metrics
    print(f"  done in {time.time()-t0:.1f}s")
    print(f"  {ubcf_metrics}")

    # --- SVD ---
    print("\n[3/4] SVD matrix factorization...")
    t0 = time.time()
    svd = SVDRecommender(n_factors=cfg['models']['svd']['n_factors'])
    svd.fit(mat)
    svd_recs = svd.recommend_all_users(cfg['data']['n_users'], train_items_idx, n=k)
    svd_recs = {u: svd_recs[u] for u in test_gt if u in svd_recs}
    svd_metrics = evaluate_rankings(svd_recs, test_gt, k)
    results['SVD'] = svd_metrics
    print(f"  done in {time.time()-t0:.1f}s")
    print(f"  {svd_metrics}")

    # --- NeuMF ---
    print("\n[4/4] NeuMF (Neural CF)...")
    t0 = time.time()
    try:
        import torch
        from src.models.ncf import NeuMF, NCFDataset, build_negative_samples, train_ncf_model, predict_topk_ncf

        ncf_cfg = cfg['models']['ncf']
        users_arr, items_arr, labels_arr = build_negative_samples(
            train, n_items=cfg['data']['n_items'], neg_ratio=ncf_cfg['neg_ratio'])

        dataset = NCFDataset(users_arr, items_arr, labels_arr)
        loader = torch.utils.data.DataLoader(dataset, batch_size=ncf_cfg['batch_size'],
                                              shuffle=True)

        model = NeuMF(cfg['data']['n_users'], cfg['data']['n_items'],
                       gmf_dim=ncf_cfg['gmf_dim'], mlp_dim=ncf_cfg['mlp_dim'],
                       mlp_hidden=ncf_cfg['mlp_hidden'])

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = train_ncf_model(model, loader, epochs=ncf_cfg['epochs'],
                                 lr=ncf_cfg['lr'], device=device)

        all_items = list(range(1, cfg['data']['n_items'] + 1))
        ncf_recs = {}
        for uid in test_gt:
            exclude = train_user_items.get(uid, set())
            ncf_recs[uid] = predict_topk_ncf(model, uid, all_items, exclude, k=k, device=device)

        ncf_metrics = evaluate_rankings(ncf_recs, test_gt, k)
        results['NeuMF'] = ncf_metrics
        print(f"  done in {time.time()-t0:.1f}s")
        print(f"  {ncf_metrics}")
    except ImportError:
        print("  PyTorch not available, skipping NeuMF")

    # --- Summary ---
    print("\n" + "="*60)
    print(f"Results @ K={k}")
    print("="*60)
    header = f"{'Model':<15} {'HR':>8} {'NDCG':>8} {'Prec':>8} {'Recall':>8}"
    print(header)
    print("-"*60)
    for name, m in results.items():
        hr = m[f'HR@{k}']
        ndcg = m[f'NDCG@{k}']
        prec = m[f'Precision@{k}']
        rec = m[f'Recall@{k}']
        print(f"{name:<15} {hr:>8.4f} {ndcg:>8.4f} {prec:>8.4f} {rec:>8.4f}")


if __name__ == '__main__':
    main()
