import random
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_fig(fig, name, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, name), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_rating_dist(ratings, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ratings['rating'].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_xlabel('Rating')
    ax.set_ylabel('Count')
    ax.set_title('Rating Distribution')
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_long_tail(counts, xlabel, title, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    sorted_counts = sorted(counts, reverse=True)
    ax.plot(range(len(sorted_counts)), sorted_counts)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.set_yscale('log')
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
