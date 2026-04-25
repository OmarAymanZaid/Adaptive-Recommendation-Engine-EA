import random


# -----------------------------
# One-Point Crossover
# -----------------------------
def one_point_crossover(vec1, vec2):
    """
    Splits both parents at one point and swaps tails.
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of same length")

    point = random.randint(1, len(vec1) - 1)

    child1 = list(vec1[:point]) + list(vec2[point:])
    child2 = list(vec2[:point]) + list(vec1[point:])

    return child1, child2


# -----------------------------
# Uniform Crossover
# -----------------------------
def uniform_crossover(vec1, vec2, swap_prob=0.5):
    """
    Each gene is swapped independently with a given probability.
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of same length")

    child1 = []
    child2 = []

    for g1, g2 in zip(vec1, vec2):
        if random.random() < swap_prob:
            child1.append(g2)
            child2.append(g1)
        else:
            child1.append(g1)
            child2.append(g2)

    return child1, child2


# -----------------------------
# Crossover Dispatcher
# -----------------------------
def crossover(parent1, parent2, method="one_point", crossover_rate=0.8, **kwargs):
    """
    Applies crossover and returns two offspring individuals.
    """

    # No crossover → return copies
    if random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()

    vec1 = parent1.vector
    vec2 = parent2.vector

    if method == "one_point":
        child_vec1, child_vec2 = one_point_crossover(vec1, vec2)

    elif method == "uniform":
        child_vec1, child_vec2 = uniform_crossover(
            vec1,
            vec2,
            swap_prob=kwargs.get("swap_prob", 0.5)
        )

    else:
        raise ValueError(f"Unknown crossover method: {method}")

    # Create new individuals (same class as parents)
    child1 = parent1.copy()
    child2 = parent2.copy()

    child1.vector = child_vec1
    child2.vector = child_vec2

    child1.fitness = None
    child2.fitness = None

    return child1, child2