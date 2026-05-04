import numpy as np
from models.user import UserIndividual
from models.item import ItemIndividual
from models.SolutionForGA import SolutionIndividual #### New import for the solution individual For GA.


def random_uniform_vector(dim, low=-1.0, high=1.0):
    return np.random.uniform(low, high, dim)

def gaussian_vector(dim, mean=0.0, std=0.1):
    return np.random.normal(mean, std, dim)

def create_user_population(user_ids, dim, init_method="uniform"):
    population = []
    
    for uid in user_ids:
        if init_method == "uniform":
            vec = random_uniform_vector(dim)
        else:
            vec = gaussian_vector(dim)
            
        population.append(UserIndividual(vec, user_id=uid))
        
    return population

def create_item_population(item_ids, dim, init_method="uniform"):
    population = []
    
    for iid in item_ids:
        if init_method == "uniform":
            vec = random_uniform_vector(dim)
        else:
            vec = gaussian_vector(dim)
            
        population.append(ItemIndividual(vec, item_id=iid))
        
    return population

def initialize_populations(user_ids, item_ids, dim, init_method="uniform"):
    # Pass the lists of IDs directly
    users = create_user_population(user_ids, dim, init_method)
    items = create_item_population(item_ids, dim, init_method)
    
    return users, items

#Under this is new part added for GA .


# --- New Function For GA ------

def create_solution_population(user_ids, item_ids, dim, pop_size, init_method="uniform"):
    population = []
    # Total genes = (number of users + number of items) * vector dimension
    total_elements = (len(user_ids) + len(item_ids)) * dim
    
    for _ in range(pop_size):
        if init_method == "uniform":
            vec = random_uniform_vector(total_elements)
        else:
            vec = gaussian_vector(total_elements)
            
        population.append(SolutionIndividual(vec, user_ids, item_ids, dim))
        
    return population