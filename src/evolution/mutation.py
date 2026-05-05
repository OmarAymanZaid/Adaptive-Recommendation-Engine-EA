import numpy as np
import random

# -----------------------------------------
# Gaussian Mutation (Non-uniform Mutation)
# -----------------------------------------
def gaussian_mutation(vector, mutation_rate=0.1, mean=0.0, std=0.1):
    """
    Vectorized Gaussian mutation.
    """
    new_vector = vector.copy()
    
    # Create a mask for which genes to mutate
    mask = np.random.rand(len(vector)) < mutation_rate
    
    # Generate noise for all genes, but only apply it where the mask is True
    noise = np.random.normal(mean, std, len(vector))
    new_vector[mask] += noise[mask]

    return new_vector

# -----------------------------------------
# Random Reset Mutation (Uniform Mutation)
# -----------------------------------------
def random_reset_mutation(vector, mutation_rate=0.1, low=-1.0, high=1.0):
    """
    Vectorized random reset mutation.
    """
    new_vector = vector.copy()
    
    mask = np.random.rand(len(vector)) < mutation_rate
    random_vals = np.random.uniform(low, high, len(vector))
    new_vector[mask] = random_vals[mask]

    return new_vector

# -----------------------------
# Mutation Dispatcher
# -----------------------------
def mutate(individual, method="gaussian", mutation_rate=0.1, **kwargs):
    vector = individual.vector

    if method == "gaussian":
        new_vector = gaussian_mutation(vector, mutation_rate=mutation_rate, mean=kwargs.get("mean", 0.0), std=kwargs.get("std", 0.1))
    elif method == "random_reset":
        new_vector = random_reset_mutation(vector, mutation_rate=mutation_rate, low=kwargs.get("low", -1.0), high=kwargs.get("high", 1.0))
    else:
        raise ValueError(f"Unknown mutation method: {method}")

    new_individual = individual.copy()
    new_individual.vector = new_vector
    new_individual.fitness = None 

    return new_individual