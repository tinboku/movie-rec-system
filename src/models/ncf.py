import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class NCFDataset(Dataset):
    def __init__(self, user_ids, item_ids, labels):
        self.users = torch.LongTensor(user_ids)
        self.items = torch.LongTensor(item_ids)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


def build_negative_samples(train_ratings, n_items=1682, neg_ratio=4, seed=42):
    """for each positive interaction, sample neg_ratio negative items"""
    rng = np.random.RandomState(seed)

    user_pos = {}
    for _, row in train_ratings.iterrows():
        uid = int(row['user_id'])
        iid = int(row['item_id'])
        if uid not in user_pos:
            user_pos[uid] = set()
        user_pos[uid].add(iid)

    users, items, labels = [], [], []
    all_items = set(range(1, n_items + 1))

    for uid, pos_items in user_pos.items():
        neg_pool = list(all_items - pos_items)
        for iid in pos_items:
            users.append(uid)
            items.append(iid)
            labels.append(1.0)
            # negative samples
            negs = rng.choice(neg_pool, size=min(neg_ratio, len(neg_pool)), replace=False)
            for neg_iid in negs:
                users.append(uid)
                items.append(neg_iid)
                labels.append(0.0)

    return np.array(users), np.array(items), np.array(labels)


class GMF(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users + 1, embed_dim)
        self.item_emb = nn.Embedding(n_items + 1, embed_dim)
        self.out = nn.Linear(embed_dim, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)
        x = u * i  # element-wise product
        return torch.sigmoid(self.out(x)).squeeze(-1)


class MLP(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=32, hidden_layers=[64, 32, 16]):
        super().__init__()
        self.user_emb = nn.Embedding(n_users + 1, embed_dim)
        self.item_emb = nn.Embedding(n_items + 1, embed_dim)

        layers = []
        input_dim = embed_dim * 2
        for h in hidden_layers:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.2))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)
        x = torch.cat([u, i], dim=-1)
        return torch.sigmoid(self.mlp(x)).squeeze(-1)


class NeuMF(nn.Module):
    """Neural Matrix Factorization = GMF + MLP"""
    def __init__(self, n_users, n_items, gmf_dim=32, mlp_dim=32,
                 mlp_hidden=[64, 32, 16]):
        super().__init__()
        # GMF part
        self.gmf_user_emb = nn.Embedding(n_users + 1, gmf_dim)
        self.gmf_item_emb = nn.Embedding(n_items + 1, gmf_dim)

        # MLP part
        self.mlp_user_emb = nn.Embedding(n_users + 1, mlp_dim)
        self.mlp_item_emb = nn.Embedding(n_items + 1, mlp_dim)

        layers = []
        input_dim = mlp_dim * 2
        for h in mlp_hidden:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        self.mlp_layers = nn.Sequential(*layers)

        # final prediction
        self.predict_layer = nn.Linear(gmf_dim + input_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for emb in [self.gmf_user_emb, self.gmf_item_emb,
                     self.mlp_user_emb, self.mlp_item_emb]:
            nn.init.normal_(emb.weight, std=0.01)
        nn.init.xavier_uniform_(self.predict_layer.weight)

    def forward(self, user, item):
        # GMF
        gmf_u = self.gmf_user_emb(user)
        gmf_i = self.gmf_item_emb(item)
        gmf_out = gmf_u * gmf_i

        # MLP
        mlp_u = self.mlp_user_emb(user)
        mlp_i = self.mlp_item_emb(item)
        mlp_input = torch.cat([mlp_u, mlp_i], dim=-1)
        mlp_out = self.mlp_layers(mlp_input)

        concat = torch.cat([gmf_out, mlp_out], dim=-1)
        return torch.sigmoid(self.predict_layer(concat)).squeeze(-1)


def train_ncf_model(model, train_loader, epochs=10, lr=0.001, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batch = 0
        for users, items, labels in train_loader:
            users = users.to(device)
            items = items.to(device)
            labels = labels.to(device)

            pred = model(users, items)
            loss = criterion(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batch += 1

        avg_loss = total_loss / n_batch
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{epochs}, loss={avg_loss:.4f}")

    return model


def predict_topk_ncf(model, user_id, all_items, exclude_items, k=10, device='cpu'):
    model.eval()
    candidates = [i for i in all_items if i not in exclude_items]
    if not candidates:
        return []

    with torch.no_grad():
        user_tensor = torch.LongTensor([user_id] * len(candidates)).to(device)
        item_tensor = torch.LongTensor(candidates).to(device)
        scores = model(user_tensor, item_tensor).cpu().numpy()

    topk_idx = np.argsort(scores)[::-1][:k]
    return [candidates[i] for i in topk_idx]
