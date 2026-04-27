import numpy as np


def predict_rating(user_vector, item_vector):
    pred = np.dot(user_vector, item_vector)

    pred = np.tanh(pred) # Squash to [-1, 1]

    pred = 1 + (pred + 1) * 2 # Map to rating scale [1, 5]

    return float(pred)


def evaluate_user(user, items_dict, user_ratings):
    """
    user: UserIndividual
    items_dict: {item_id: ItemIndividual}
    user_ratings: list of (item_id, true_rating)
    """

    if not user_ratings:
        user.fitness = -1e9  # Penalize users with no ratings
        return

    errors = []

    for item_id, true_rating in user_ratings:
        item = items_dict[item_id]

        pred = predict_rating(user.vector, item.vector)
        error = (pred - true_rating) ** 2
        errors.append(error)

    mse = np.mean(errors)

    # Higher fitness is better → use negative MSE
    user.fitness = -mse


def evaluate_item(item, users_dict, item_ratings):
    """
    item: ItemIndividual
    users_dict: {user_id: UserIndividual}
    item_ratings: list of (user_id, true_rating)
    """

    if not item_ratings:
        item.fitness = -1e9  # Penalize items with no ratings
        return

    errors = []

    for user_id, true_rating in item_ratings:
        user = users_dict[user_id]

        pred = predict_rating(user.vector, item.vector)
        error = (pred - true_rating) ** 2
        errors.append(error)

    mse = np.mean(errors)

    item.fitness = -mse


def evaluate_users(user_population, item_population, dataset):
    """
    dataset["user_ratings"]: {user_id: [(item_id, rating), ...]}
    """

    items_dict = {item.item_id: item for item in item_population}

    for user in user_population:
        user_ratings = dataset["user_ratings"].get(user.user_id, [])
        evaluate_user(user, items_dict, user_ratings)


def evaluate_items(item_population, user_population, dataset):
    """
    dataset["item_ratings"]: {item_id: [(user_id, rating), ...]}
    """

    users_dict = {user.user_id: user for user in user_population}

    for item in item_population:
        item_ratings = dataset["item_ratings"].get(item.item_id, [])
        evaluate_item(item, users_dict, item_ratings)