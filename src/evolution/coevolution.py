import copy

from evolution.evaluation import evaluate_users, evaluate_items
from evolution.selection import select_population
from evolution.crossover import crossover
from evolution.mutation import mutate
from evolution.replacement import replace_population
from evolution.population import initialize_populations

def run_coevolution(dataset, config):
    """
    Main cooperative coevolutionary algorithm loop.
    """

    user_ids = list(dataset["user_ratings"].keys())
    item_ids = list(dataset["item_ratings"].keys())

    users, items = initialize_populations(
        user_ids,
        item_ids,
        config["vector_dim"],
        config["init_method"]
    )

    user_history = []
    item_history = []

    # -------------------------
    # Evolution loop
    # -------------------------
    for gen in range(config["num_generations"]):

        # Evaluation
        evaluate_users(users, items, dataset)
        evaluate_items(items, users, dataset)

        # Store stats
        best_user_fitness = max(u.fitness for u in users)
        best_item_fitness = max(i.fitness for i in items)

        user_history.append(best_user_fitness)
        item_history.append(best_item_fitness)

        # -------------------------
        # Create a user mating pool
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

            c1, c2 = crossover(p1, p2, method=config["crossover_method"], crossover_rate=config["crossover_rate"])
            c1 = mutate(c1, method=config["mutation_method"], mutation_rate=config["mutation_rate"])
            c2 = mutate(c2, method=config["mutation_method"], mutation_rate=config["mutation_rate"])

            user_offspring.extend([c1, c2])
            
        user_offspring = user_offspring[:len(users)]

        # ---------------------------
        # Create an item mating pool
        # ---------------------------
        item_selected = select_population(
            items,
            method=config["selection_method"],
            tournament_size=config.get("tournament_size", 3)
        )

        item_offspring = []
        for i in range(0, len(item_selected), 2):
            p1 = item_selected[i]
            p2 = item_selected[(i + 1) % len(item_selected)]

            c1, c2 = crossover(p1, p2, method=config["crossover_method"], crossover_rate=config["crossover_rate"])
            c1 = mutate(c1, method=config["mutation_method"], mutation_rate=config["mutation_rate"])
            c2 = mutate(c2, method=config["mutation_method"], mutation_rate=config["mutation_rate"])

            item_offspring.extend([c1, c2])
            
        item_offspring = item_offspring[:len(items)]

        # -------------------------
        # Evaluate Offspring Before Replacement
        # -------------------------
        # We must score the babies so they can fairly compete against their parents!
        evaluate_users(user_offspring, items, dataset)
        evaluate_items(item_offspring, users, dataset)

        # -------------------------
        # Replacement
        # -------------------------
        users = replace_population(
            users,
            user_offspring,
            method=config["replacement_method"],
            elite_size=config.get("elite_size", 2)
        )

        items = replace_population(
            items,
            item_offspring,
            method=config["replacement_method"],
            elite_size=config.get("elite_size", 2)
)

        # Optional logging
        print(f"Gen {gen}: User best={best_user_fitness:.4f}, Item best={best_item_fitness:.4f}")

    return {
        "users": users,
        "items": items,
        "user_history": user_history,
        "item_history": item_history,
    }