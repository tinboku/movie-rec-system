import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pickle
import torch
from fastapi import FastAPI, HTTPException, Query

from src.models.mf import SVDRecommender
from src.models.ncf import NeuMF, predict_topk_ncf

app = FastAPI(title="Movie Recommendation API")

# globals
svd_model = None
ncf_model = None
title_map = {}
train_user_items = {}
N_USERS = 943
N_ITEMS = 1682


def load_models():
    global svd_model, ncf_model, title_map, train_user_items

    base = os.path.join(os.path.dirname(__file__), '..', 'saved_models')

    with open(os.path.join(base, 'svd.pkl'), 'rb') as f:
        svd_model = pickle.load(f)

    with open(os.path.join(base, 'titles.pkl'), 'rb') as f:
        title_map = pickle.load(f)

    with open(os.path.join(base, 'train_user_items.pkl'), 'rb') as f:
        train_user_items = pickle.load(f)

    ncf_model = NeuMF(N_USERS, N_ITEMS, gmf_dim=32, mlp_dim=32, mlp_hidden=[64, 32, 16])
    ncf_model.load_state_dict(torch.load(os.path.join(base, 'neumf.pt'), map_location='cpu'))
    ncf_model.eval()


@app.on_event("startup")
def startup():
    load_models()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = Query(default=10, le=50),
              model: str = Query(default="svd")):
    if user_id < 1 or user_id > N_USERS:
        raise HTTPException(status_code=400, detail=f"user_id must be 1-{N_USERS}")

    exclude = train_user_items.get(user_id, set())

    if model == "svd":
        uid_idx = user_id - 1
        exclude_idx = {i - 1 for i in exclude}
        recs = svd_model.recommend(uid_idx, n=k, exclude_items=exclude_idx)
        item_ids = [r + 1 for r in recs]
    elif model == "ncf":
        all_items = list(range(1, N_ITEMS + 1))
        item_ids = predict_topk_ncf(ncf_model, user_id, all_items, exclude, k=k)
    else:
        raise HTTPException(status_code=400, detail="model must be 'svd' or 'ncf'")

    results = []
    for iid in item_ids:
        results.append({
            "item_id": iid,
            "title": title_map.get(iid, "Unknown"),
        })

    return {
        "user_id": user_id,
        "model": model,
        "recommendations": results,
    }
