import random
import numpy as np


def tournament_selection(population, tournament_size=3):
    """
    Select one individual using tournament selection.
    Higher fitness wins.
    """

    contenders = random.sample(population, tournament_size)
    best = max(contenders, key=lambda ind: ind.fitness)

    return best



def roulette_selection(population):
    """
    Fitness-proportionate selection.
    Handles negative fitness by shifting values.
    """

    fitnesses = np.array([ind.fitness for ind in population])

    # Shift if needed (important because we use negative MSE)
    min_fit = np.min(fitnesses)
    if min_fit < 0:
        fitnesses = fitnesses - min_fit + 1e-6  # avoid zero

    total = np.sum(fitnesses)

    if total == 0:
        return random.choice(population)

    probs = fitnesses / total

    return np.random.choice(population, p=probs)


def select_parents(population, method="tournament", tournament_size=3):
    """
    Returns two parents for crossover.
    """

    if method == "tournament":
        p1 = tournament_selection(population, tournament_size)
        p2 = tournament_selection(population, tournament_size)

    elif method == "roulette":
        p1 = roulette_selection(population)
        p2 = roulette_selection(population)

    else:
        raise ValueError(f"Unknown selection method: {method}")

    return p1, 


def select_population(population, method="tournament", size=None, tournament_size=3):
    """
    Creates a mating pool of selected individuals.
    """

    if size is None:
        size = len(population)

    selected = []

    for _ in range(size):
        if method == "tournament":
            selected.append(tournament_selection(population, tournament_size))
        elif method == "roulette":
            selected.append(roulette_selection(population))
        else:
            raise ValueError(f"Unknown selection method: {method}")

    return selected