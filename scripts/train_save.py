"""train models and save to disk for serving"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pickle
import numpy as np
import torch

from src.data_loader import load_ratings, load_items, train_test_split_by_time, build_interaction_matrix
from src.models.mf import SVDRecommender
from src.models.ncf import NeuMF, NCFDataset, build_negative_samples, train_ncf_model
from src.utils import set_seed


def main():
    set_seed(42)
    os.makedirs('saved_models', exist_ok=True)

    ratings = load_ratings('data/raw')
    items = load_items('data/raw')
    train, test = train_test_split_by_time(ratings, test_ratio=0.2)

    # item title lookup
    title_map = dict(zip(items['item_id'], items['title']))
    with open('saved_models/titles.pkl', 'wb') as f:
        pickle.dump(title_map, f)

    # train user items (for excluding at inference)
    train_user_items = {}
    for uid, group in train.groupby('user_id'):
        train_user_items[uid] = set(group['item_id'].values)
    with open('saved_models/train_user_items.pkl', 'wb') as f:
        pickle.dump(train_user_items, f)

    mat = build_interaction_matrix(train, 943, 1682)

    # --- SVD ---
    print('training SVD...')
    svd = SVDRecommender(n_factors=50)
    svd.fit(mat)
    with open('saved_models/svd.pkl', 'wb') as f:
        pickle.dump(svd, f)
    print('  saved saved_models/svd.pkl')

    # --- NeuMF ---
    print('training NeuMF...')
    users_arr, items_arr, labels_arr = build_negative_samples(
        train, n_items=1682, neg_ratio=4, seed=42)
    dataset = NCFDataset(users_arr, items_arr, labels_arr)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    model = NeuMF(943, 1682, gmf_dim=32, mlp_dim=32, mlp_hidden=[64, 32, 16])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_ncf_model(model, loader, epochs=15, lr=0.001, device=device)
    torch.save(model.state_dict(), 'saved_models/neumf.pt')
    print('  saved saved_models/neumf.pt')

    print('done.')


if __name__ == '__main__':
    main()
