import numpy as np

# -----------------------------------------
# Uniform Mutation (Random Reset Mutation)
# -----------------------------------------
def uniform_mutation(vector, mutation_rate=0.1, low=-1.0, high=1.0):
    """
    Vectorized random reset mutation.
    """
    new_vector = vector.copy()
    
    mask = np.random.rand(len(vector)) < mutation_rate
    random_vals = np.random.uniform(low, high, len(vector))
    new_vector[mask] = random_vals[mask]

    return new_vector

# -----------------------------------------
# Non-uniform Mutation (Gaussian Mutation)
# -----------------------------------------
def non_uniform_mutation(vector, mutation_rate=0.1, mean=0.0, std=0.1):
    """
    Vectorized non-uniform mutation using Gaussian noise.
    """
    new_vector = vector.copy()
    
    # Create a mask for which genes to mutate
    mask = np.random.rand(len(vector)) < mutation_rate
    
    # Generate noise for all genes, but only apply it where the mask is True
    noise = np.random.normal(mean, std, len(vector))
    new_vector[mask] += noise[mask]

    return new_vector

# -----------------------------
# Mutation Dispatcher
# -----------------------------
def mutate(individual, method="non_uniform", mutation_rate=0.1, **kwargs):
    vector = individual.vector

    if method == "uniform":
        new_vector = uniform_mutation(vector, mutation_rate=mutation_rate, low=kwargs.get("low", -1.0), high=kwargs.get("high", 1.0))
    elif method == "non_uniform":
        new_vector = non_uniform_mutation(vector, mutation_rate=mutation_rate, mean=kwargs.get("mean", 0.0), std=kwargs.get("std", 0.1))
    else:
        raise ValueError(f"Unknown mutation method: {method}")

    new_individual = individual.copy()
    new_individual.vector = new_vector
    new_individual.fitness = None 

    return new_individual