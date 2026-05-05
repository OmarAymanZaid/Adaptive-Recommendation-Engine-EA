import copy

def generational_replacement(offspring):
    return offspring

def elitist_replacement(parents, offspring, elite_size=2):
    sorted_parents = sorted(
        parents, 
        key=lambda ind: ind.fitness if ind.fitness is not None else -float('inf'), 
        reverse=True
    )
    elites = [p.copy() for p in sorted_parents[:elite_size]]
    new_population = elites + offspring
    return new_population[:len(parents)]

def species_preserving_replacement(parents, offspring):
    """
    Ensures that the absolute best version of EVERY unique ID survives.
    Prevents species extinction (Genetic Drift).
    """
    combined = parents + offspring
    best_by_id = {}
    
    for ind in combined:
        # Dynamically grab the ID whether this is a User or an Item
        ind_id = getattr(ind, 'user_id', getattr(ind, 'item_id', None))
        fit = ind.fitness if ind.fitness is not None else -float('inf')
        
        # Check if we already have a saved version of this ID
        current_best = best_by_id.get(ind_id)
        current_best_fit = (current_best.fitness 
                            if current_best and current_best.fitness is not None 
                            else -float('inf'))
        
        # If we haven't seen this ID yet, or this new one is fitter, save it!
        if ind_id not in best_by_id or fit > current_best_fit:
            best_by_id[ind_id] = ind.copy()
            
    # Return a list containing exactly one elite representative for every ID
    return list(best_by_id.values())


def replace_population(parents, offspring, method="species", elite_size=2):
    if method == "generational":
        return generational_replacement(offspring)

    elif method == "elitist":
        return elitist_replacement(parents, offspring, elite_size)

    elif method == "species":
        return species_preserving_replacement(parents, offspring)

    else:
        raise ValueError(f"Unknown replacement method: {method}")