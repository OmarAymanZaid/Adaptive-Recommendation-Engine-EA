import numpy as np
import random


# -----------------------------
# Gaussian Mutation
# -----------------------------
def gaussian_mutation(vector, mutation_rate=0.1, mean=0.0, std=0.1):
    """
    Adds Gaussian noise to each gene with a given probability.
    """
    new_vector = vector.copy()

    for i in range(len(new_vector)):
        if random.random() < mutation_rate:
            new_vector[i] += np.random.normal(mean, std)

    return new_vector


# -----------------------------
# Random Reset Mutation
# -----------------------------
def random_reset_mutation(vector, mutation_rate=0.1, low=-1.0, high=1.0):
    """
    Replaces gene values randomly within a range.
    """
    new_vector = vector.copy()

    for i in range(len(new_vector)):
        if random.random() < mutation_rate:
            new_vector[i] = np.random.uniform(low, high)

    return new_vector


# -----------------------------
# Mutation Dispatcher
# -----------------------------
def mutate(individual, method="gaussian", mutation_rate=0.1, **kwargs):
    """
    Applies mutation to an individual and returns a new mutated individual.
    """

    vector = individual.vector

    if method == "gaussian":
        new_vector = gaussian_mutation(
            vector,
            mutation_rate=mutation_rate,
            mean=kwargs.get("mean", 0.0),
            std=kwargs.get("std", 0.1),
        )

    elif method == "random_reset":
        new_vector = random_reset_mutation(
            vector,
            mutation_rate=mutation_rate,
            low=kwargs.get("low", -1.0),
            high=kwargs.get("high", 1.0),
        )

    else:
        raise ValueError(f"Unknown mutation method: {method}")

    # Create new individual (important: don't modify original)
    new_individual = individual.copy()
    new_individual.vector = new_vector
    new_individual.fitness = None  # fitness must be recomputed

    return new_individual