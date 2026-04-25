import copy

from evolution.evaluation import evaluate_users, evaluate_items
from evolution.selection import select_population
from evolution.crossover import crossover
from evolution.mutation import mutate
from evolution.replacement import elitist_replacement, generational_replacement
from evolution.population import initialize_populations

def run_coevolution(
    dataset,
    config,
):
    """
    Main coevolutionary algorithm loop.
    """

    # -------------------------
    # 1. Initialize populations
    # -------------------------
    users, items = initialize_populations(
        config["user_population_size"],
        config["item_population_size"],
        config["vector_dim"],
        config["init_method"]
    )

    user_history = []
    item_history = []

    # -------------------------
    # 2. Evolution loop
    # -------------------------
    for gen in range(config["num_generations"]):

        # -------------------------
        # Evaluation (co-dependent)
        # -------------------------
        evaluate_users(users, items, dataset)
        evaluate_items(items, users, dataset)

        # -------------------------
        # Store stats
        # -------------------------
        best_user_fitness = max(u.fitness for u in users)
        best_item_fitness = max(i.fitness for i in items)

        user_history.append(best_user_fitness)
        item_history.append(best_item_fitness)

        # -------------------------
        # Create new user population
        # -------------------------
        user_selected = select_population(
            users,
            method=config["selection_method"],
            tournament_size=config.get("tournament_size", 3)
        )

        user_offspring = []

        for i in range(0, len(user_selected), 2):
            p1 = user_selected[i]
            p2 = user_selected[(i + 1) % len(user_selected)]

            c1, c2 = crossover(
                p1,
                p2,
                method=config["crossover_method"],
                crossover_rate=config["crossover_rate"]
            )

            c1 = mutate(
                c1,
                method=config["mutation_method"],
                mutation_rate=config["mutation_rate"]
            )

            c2 = mutate(
                c2,
                method=config["mutation_method"],
                mutation_rate=config["mutation_rate"]
            )

            user_offspring.extend([c1, c2])

        # -------------------------
        # Create new item population
        # -------------------------
        item_selected = select_population(
            items,
            method=config["selection_method"],
            tournament_size=config.get("tournament_size", 3)
        )

        item_offspring = []

        for i in range(0, len(item_selected), 2):
            p1 = item_selected[i]
            p2 = item_selected[(i + 1) % len(item_selected)]

            c1, c2 = crossover(
                p1,
                p2,
                method=config["crossover_method"],
                crossover_rate=config["crossover_rate"]
            )

            c1 = mutate(
                c1,
                method=config["mutation_method"],
                mutation_rate=config["mutation_rate"]
            )

            c2 = mutate(
                c2,
                method=config["mutation_method"],
                mutation_rate=config["mutation_rate"]
            )

            item_offspring.extend([c1, c2])

        # -------------------------
        # Replacement
        # -------------------------
        if config["replacement"] == "elitist":
            users = elitist_replacement(users, user_offspring, elite_size=2)
            items = elitist_replacement(items, item_offspring, elite_size=2)

        else:
            users = generational_replacement(user_offspring)
            items = generational_replacement(item_offspring)

        # -------------------------
        # Optional logging
        # -------------------------
        print(f"Gen {gen}: User best={best_user_fitness:.4f}, Item best={best_item_fitness:.4f}")

    # -------------------------
    # Return final state
    # -------------------------
    return {
        "users": users,
        "items": items,
        "user_history": user_history,
        "item_history": item_history,
    }