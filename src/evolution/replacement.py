import copy

def generational_replacement(offspring):
    """
    Full replacement: next generation = offspring only
    """
    return offspring


def elitist_replacement(parents, offspring, elite_size=2):
    """
    Keeps top-performing individuals from parents,
    fills the rest with offspring.
    """

    # Sort parents by fitness (descending because higher is better)
    sorted_parents = sorted(parents, key=lambda ind: ind.fitness, reverse=True)

    elites = [p.copy() for p in sorted_parents[:elite_size]]

    # Fill remaining slots
    new_population = elites + offspring

    # Trim if oversized
    return new_population[:len(parents)]

