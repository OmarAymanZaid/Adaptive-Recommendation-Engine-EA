PARAMS = {
    "population_size": 50,
    "num_generations": 100,
    "vector_dim": 10,

    "selection_method": "tournament",
    "tournament_size": 2,      # Needed for tournament selection

    "crossover_method": "one_point",
    "mutation_method": "gaussian",

    "mutation_rate": 0.3,
    "crossover_rate": 0.8,

    "init_method": "uniform",
    
    "replacement_method": "species",
    "elite_size": 2,

    "random_seed": 42,
}