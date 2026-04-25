import numpy as np
import pandas as pd
import os


def generate_synthetic_data(
    num_users=20,
    num_items=30,
    dim=10,
    rating_scale=(1, 5),
    noise=0.1,
    sparsity=0.3,
    seed=42
):
    np.random.seed(seed)

    # True latent vectors (ground truth)
    true_users = np.random.uniform(-1, 1, (num_users, dim))
    true_items = np.random.uniform(-1, 1, (num_items, dim))

    user_ratings = {u: [] for u in range(num_users)}
    item_ratings = {i: [] for i in range(num_items)}

    for u in range(num_users):
        for i in range(num_items):

            # sparsity (simulate missing ratings)
            if np.random.rand() > sparsity:
                continue

            rating = np.dot(true_users[u], true_items[i])
            rating += np.random.normal(0, noise)

            # normalize to rating scale
            rating = np.clip(rating, -1, 1)
            rating = rating_scale[0] + (rating + 1) * (rating_scale[1] - rating_scale[0]) / 2

            rating = float(rating)

            user_ratings[u].append((i, rating))
            item_ratings[i].append((u, rating))

    return {
        "user_ratings": user_ratings,
        "item_ratings": item_ratings,
        "num_users": num_users,
        "num_items": num_items
    }


def load_real_dataset(path):
    df = pd.read_csv(path)

    user_ratings = {}
    item_ratings = {}

    for row in df.itertuples(index=False):
        u = int(row.user_id)
        i = int(row.item_id)
        r = float(row.rating)

        if u not in user_ratings:
            user_ratings[u] = []
        if i not in item_ratings:
            item_ratings[i] = []

        user_ratings[u].append((i, r))
        item_ratings[i].append((u, r))

    return {
        "user_ratings": user_ratings,
        "item_ratings": item_ratings,
        "num_users": len(user_ratings),
        "num_items": len(item_ratings)
    }


def load_dataset(
    mode="synthetic",
    path=None,
    **kwargs
):
    """
    mode:
        - synthetic
        - real
    """

    if mode == "synthetic":
        return generate_synthetic_data(**kwargs)

    elif mode == "real":
        if path is None:
            raise ValueError("Path required for real dataset")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")

        return load_real_dataset(path)

    else:
        raise ValueError("Unknown dataset mode")