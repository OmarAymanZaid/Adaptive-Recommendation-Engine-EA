import numpy as np
from models.user import UserIndividual
from models.item import ItemIndividual


def random_uniform_vector(dim, low=-1.0, high=1.0):
    return np.random.uniform(low, high, dim)


def gaussian_vector(dim, mean=0.0, std=0.1):
    return np.random.normal(mean, std, dim)


def create_user_population(size, dim, init_method="uniform"):
    population = []

    for i in range(size):
        if init_method == "uniform":
            vec = random_uniform_vector(dim)
        else:
            vec = gaussian_vector(dim)

        population.append(UserIndividual(vec, user_id=i))

    return population


def create_item_population(size, dim, init_method="uniform"):
    population = []

    for i in range(size):
        if init_method == "uniform":
            vec = random_uniform_vector(dim)
        else:
            vec = gaussian_vector(dim)

        population.append(ItemIndividual(vec, item_id=i))

    return population


def initialize_populations(user_size, item_size, dim, init_method="uniform"):
    users = create_user_population(user_size, dim, init_method)
    items = create_item_population(item_size, dim, init_method)

    return users, items