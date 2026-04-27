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
    
    sorted_parents = sorted(
        parents, 
        key=lambda ind: ind.fitness if ind.fitness is not None else -float('inf'), 
        reverse=True
    )

    elites = [p.copy() for p in sorted_parents[:elite_size]]

    new_population = elites + offspring

    return new_population[:len(parents)]