import numpy as np
import random

# -----------------------------
# One-Point Crossover
# -----------------------------
def one_point_crossover(vec1, vec2):
    """
    Splits both parents at one point and swaps tails using NumPy slicing.
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of same length")

    point = random.randint(1, len(vec1) - 1)

    child1 = np.concatenate((vec1[:point], vec2[point:]))
    child2 = np.concatenate((vec2[:point], vec1[point:]))

    return child1, child2

# -----------------------------
# Uniform Crossover
# -----------------------------
def uniform_crossover(vec1, vec2, swap_prob=0.5):
    """
    Vectorized uniform crossover using a boolean mask.
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of same length")

    # Generate a boolean mask for the entire vector at once
    mask = np.random.rand(len(vec1)) < swap_prob
    
    # np.where(condition, true_array, false_array)
    child1 = np.where(mask, vec2, vec1)
    child2 = np.where(mask, vec1, vec2)

    return child1, child2

# -----------------------------
# Crossover Dispatcher
# -----------------------------
def crossover(parent1, parent2, method="one_point", crossover_rate=0.8, **kwargs):
    if random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()

    vec1 = parent1.vector
    vec2 = parent2.vector

    if method == "one_point":
        child_vec1, child_vec2 = one_point_crossover(vec1, vec2)
    elif method == "uniform":
        child_vec1, child_vec2 = uniform_crossover(vec1, vec2, swap_prob=kwargs.get("swap_prob", 0.5))
    else:
        raise ValueError(f"Unknown crossover method: {method}")

    child1 = parent1.copy()
    child2 = parent2.copy()

    child1.vector = child_vec1
    child2.vector = child_vec2
    child1.fitness = None
    child2.fitness = None

    return child1, child2