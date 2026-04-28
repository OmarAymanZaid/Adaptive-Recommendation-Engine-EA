import numpy as np
import math

def get_predictions_and_truths(users, items, dataset):
    """
    Helper function to safely extract all valid predictions and 
    their corresponding true ratings from the dataset.
    """
    users_dict = {u.user_id: u for u in users}
    items_dict = {i.item_id: i for i in items}
    
    preds = []
    truths = []
    
    # Iterate through the true ratings in the dataset
    for user_id, ratings in dataset["user_ratings"].items():
        user = users_dict.get(user_id)
        if user is None:
            continue
            
        for item_id, true_rating in ratings:
            item = items_dict.get(item_id)
            if item is None:
                continue
                
            # Use the .predict() method from BaseIndividual
            pred = user.predict(item.vector)
            preds.append(pred)
            truths.append(true_rating)
            
    return np.array(preds), np.array(truths)

def calculate_rmse(users, items, dataset):

    preds, truths = get_predictions_and_truths(users, items, dataset)
    if len(preds) == 0:
        return float('inf')
        
    mse = np.mean((preds - truths) ** 2)
    return math.sqrt(mse)

def calculate_mae(users, items, dataset):
    
    preds, truths = get_predictions_and_truths(users, items, dataset)
    if len(preds) == 0:
        return float('inf')
        
    return np.mean(np.abs(preds - truths))