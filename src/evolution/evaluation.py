import numpy as np

def predict_rating(user_vector, item_vector):
    return float(np.dot(user_vector, item_vector))

def evaluate_user(user, items_dict, user_ratings):
    if not user_ratings:
        user.fitness = -float('inf') # Worst possible fitness
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
        item.fitness = -float('inf') # Worst possible fitness
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


def evaluate_users(user_population, item_population, dataset):

    items_dict = {item.item_id: item for item in item_population}

    for user in user_population:
        user_ratings = dataset["user_ratings"].get(user.user_id, [])
        evaluate_user(user, items_dict, user_ratings)


def evaluate_items(item_population, user_population, dataset):
    
    users_dict = {user.user_id: user for user in user_population}

    for item in item_population:
        item_ratings = dataset["item_ratings"].get(item.item_id, [])
        evaluate_item(item, users_dict, item_ratings)



#Under this is new part added for GA .


# --- New Function For GA ------

def evaluate_single_solution(solution, dataset):
    errors = []
    
    # Loop through all real user ratings to compute overall MSE
    for user_id, ratings in dataset.get("user_ratings", {}).items():
        if not ratings:
            continue
            
        u_vec = solution.get_user_vector(user_id)
        if u_vec is None: 
            continue
            
        for item_id, true_rating in ratings:
            i_vec = solution.get_item_vector(item_id)
            if i_vec is None:
                continue 
                
            pred = float(np.dot(u_vec, i_vec))
            error = (pred - true_rating) ** 2
            errors.append(error)

    if not errors:
        solution.fitness = -float('inf') 
    else:
        mse = np.mean(errors)
        solution.fitness = -mse

def evaluate_solutions(solution_population, dataset):
    for solution in solution_population:
        evaluate_single_solution(solution, dataset)