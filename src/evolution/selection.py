import random
import numpy as np


# Tournament Selection
def tournament_selection(population, tournament_size=3):
    contenders = random.sample(population, tournament_size)
    best = max(contenders, key=lambda ind: ind.fitness)
    return best


# Roulette Selection
def roulette_selection(population, size=1):
    fitnesses = np.array([ind.fitness for ind in population])

    valid_mask = np.isfinite(fitnesses)
    
    if not np.any(valid_mask):
        return np.random.choice(population, size=size).tolist()

    min_fit = np.min(fitnesses[valid_mask])
    
    shifted_fitnesses = np.zeros_like(fitnesses)
    if min_fit < 0:
        shifted_fitnesses[valid_mask] = fitnesses[valid_mask] - min_fit + 1e-6 
    else:
        shifted_fitnesses[valid_mask] = fitnesses[valid_mask]

    total = np.sum(shifted_fitnesses)

    if total == 0:
        return np.random.choice(population, size=size).tolist()

    probs = shifted_fitnesses / total

    selected = np.random.choice(population, size=size, p=probs)
    
    return selected[0] if size == 1 else selected.tolist()


# Parent Selection Dispatcher
def select_parents(population, method="tournament", tournament_size=3):
    if method == "tournament":
        p1 = tournament_selection(population, tournament_size)
        p2 = tournament_selection(population, tournament_size)
    elif method == "roulette":
        parents = roulette_selection(population, size=2)
        p1, p2 = parents[0], parents[1]
    else:
        raise ValueError(f"Unknown selection method: {method}")

    return p1, p2 


# Population Selection Dispatcher
def select_population(population, method="tournament", size=None, tournament_size=3):
    if size is None:
        size = len(population)

    selected = []

    if method == "tournament":
        for _ in range(size):
            selected.append(tournament_selection(population, tournament_size))
    elif method == "roulette":
        selected = roulette_selection(population, size=size)
    else:
        raise ValueError(f"Unknown selection method: {method}")

    return selected