# MovieLens Recommendation System

Recommendation system built on MovieLens 100K dataset, comparing collaborative filtering approaches from traditional methods to neural models.

## Models

- **Popularity baseline** - recommend most popular items
- **User-based / Item-based CF** - KNN with cosine similarity
- **SVD** - matrix factorization via truncated SVD
- **NeuMF** - Neural Collaborative Filtering (GMF + MLP), PyTorch

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# full experiment
python run_experiment.py

# tests
pytest tests/ -v
```

## Project Structure

- `src/` - data loading, metrics, models
- `notebooks/` - EDA + experiment notebooks
- `configs/` - hyperparameters
- `data/raw/` - MovieLens 100K (u.data, u.item, u.user)

## Data

MovieLens 100K: 100,000 ratings from 943 users on 1,682 movies.
Density ~6.3%.
