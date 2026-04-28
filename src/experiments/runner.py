import copy
import time
from config.parameters import PARAMS
from utils.seeds import set_seed
from dataloader.loader import load_dataset
from evolution.coevolution import run_coevolution
from utils.metrics import calculate_rmse, calculate_mae

def run_experiment(name, overrides, dataset):
    """
    Runs a single evolutionary experiment with modified parameters.
    """
    print(f"\n--- Running Experiment: {name} ---")
    
    # Clone the default params and apply the overrides
    config = copy.deepcopy(PARAMS)
    config.update(overrides)
    
    # Ensure a fresh seed for an honest comparison
    set_seed(config["random_seed"])
    
    # Time the execution
    start_time = time.time()
    results = run_coevolution(dataset, config)
    execution_time = time.time() - start_time
    
    # Calculate final metrics
    final_users = results["users"]
    final_items = results["items"]
    
    rmse = calculate_rmse(final_users, final_items, dataset)
    mae = calculate_mae(final_users, final_items, dataset)
    
    return {
        "name": name,
        "rmse": rmse,
        "mae": mae,
        "time": execution_time
    }

def run_all_experiments():
    """
    Master function to define and execute the comparison batch.
    """
    print("Loading dataset for experiments...")
    dataset = load_dataset() # Uses the synthetic dataset generator
    
    # Define the strategies you want to compare for your report
    experiments = [
        {"name": "Baseline (One-Point/Tourn)", "overrides": {}},
        {"name": "Uniform Crossover", "overrides": {"crossover_method": "uniform"}},
        {"name": "Roulette Selection", "overrides": {"selection_method": "roulette"}},
        {"name": "High Mutation (30%)", "overrides": {"mutation_rate": 0.30}},
        {"name": "Large Population (100)", "overrides": {"population_size": 100}}
    ]
    
    # Run them and collect results
    results = []
    for exp in experiments:
        res = run_experiment(exp["name"], exp["overrides"], dataset)
        results.append(res)
        
    # Print the final comparison table
    print("\n" + "="*60)
    print("                 EXPERIMENT COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Strategy':<30} | {'RMSE':<8} | {'MAE':<8} | {'Time (s)':<8}")
    print("-" * 60)
    for res in results:
        print(f"{res['name']:<30} | {res['rmse']:<8.4f} | {res['mae']:<8.4f} | {res['time']:<8.2f}")
    print("="*60)

if __name__ == "__main__":
    run_all_experiments()