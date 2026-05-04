import copy
import numpy as np

from evolution.evaluation import evaluate_solutions
from evolution.selection import select_population
from evolution.crossover import crossover
from evolution.mutation import mutate
from evolution.replacement import elitist_replacement
from evolution.population import create_solution_population

def run_standard_ga(dataset, config):
    """
    Main standard genetic algorithm loop (Baseline comparison).
    """

    user_ids = list(dataset["user_ratings"].keys())
    item_ids = list(dataset["item_ratings"].keys())
    
    # A standard GA needs an explicit population size config
    pop_size = config.get("population_size", 50)

    # Initialize a single population of complete solutions
    population = create_solution_population(
        user_ids,
        item_ids,
        config["vector_dim"],
        pop_size,
        config["init_method"]
    )

    history = []

    # -------------------------
    # Evolution loop
    # -------------------------
    for gen in range(config["num_generations"]):

        # 1. Evaluation
        evaluate_solutions(population, dataset)

        # Store stats
        best_fitness = max(sol.fitness for sol in population)
        history.append(best_fitness)

        # 2. Selection
        selected = select_population(
            population,
            method=config["selection_method"],
            size=pop_size,
            tournament_size=config.get("tournament_size", 3)
        )

        # 3. Crossover & Mutation
        offspring = []
        for i in range(0, len(selected), 2):
            p1 = selected[i]
            p2 = selected[(i + 1) % len(selected)]

            c1, c2 = crossover(p1, p2, method=config["crossover_method"], crossover_rate=config["crossover_rate"])
            c1 = mutate(c1, method=config["mutation_method"], mutation_rate=config["mutation_rate"])
            c2 = mutate(c2, method=config["mutation_method"], mutation_rate=config["mutation_rate"])

            offspring.extend([c1, c2])
            
        # Ensure offspring matches exact pop size
        offspring = offspring[:pop_size]

        # 4. Evaluate Offspring Before Replacement
        evaluate_solutions(offspring, dataset)

        # 5. Replacement
        # We use strict elitist replacement here
        population = elitist_replacement(population, offspring, elite_size=config.get("elite_size", 2))

        # Optional logging
        print(f"Gen {gen}: Best Fitness (MSE)={best_fitness:.4f}")

    # Identify the absolute best solution found across the final population
    best_solution = max(population, key=lambda ind: ind.fitness)

    return {
        "population": population,
        "best_solution": best_solution,
        "history": history,
    }