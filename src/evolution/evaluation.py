import numpy as np

def predict_rating(user_vector, item_vector):
    return float(np.dot(user_vector, item_vector))

def evaluate_user(user, items_dict, user_ratings):
    if not user_ratings:
        user.fitness = -float('inf') 
        return

    errors = []
    for item_id, true_rating in user_ratings:
        item = items_dict.get(item_id)
        if item is None:
            continue 
        pred = predict_rating(user.vector, item.vector)
        error = (pred - true_rating) ** 2
        errors.append(error)

    if not errors:
        user.fitness = -float('inf') 
        return

    mse = np.mean(errors)
    user.fitness = -mse


def evaluate_item(item, users_dict, item_ratings):
    if not item_ratings:
        item.fitness = -float('inf') 
        return

    errors = []
    for user_id, true_rating in item_ratings:
        user = users_dict.get(user_id)
        if user is None:
            continue 

        pred = predict_rating(user.vector, item.vector)
        error = (pred - true_rating) ** 2
        errors.append(error)

    if not errors:
        item.fitness = -float('inf')
        return

    mse = np.mean(errors)
    item.fitness = -mse