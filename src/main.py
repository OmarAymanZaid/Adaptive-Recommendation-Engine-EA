import numpy as np
from config.parameters import PARAMS
from utils.seeds import set_seed
from dataloader.loader import load_dataset
from evolution.coevolution import run_coevolution

def main():

    # -------------------------
    # 1. Set seed (reproducibility)
    # -------------------------
    set_seed(PARAMS["random_seed"])

    # -------------------------
    # 2. Load dataset
    # -------------------------
    # Using default synthetic parameters
    dataset = load_dataset()

    # -------------------------
    # 3. Run coevolution
    # -------------------------
    results = run_coevolution(
        dataset=dataset,
        config=PARAMS
    )

    # -------------------------
    # 4. Print summary
    # -------------------------
    print("\n===== FINAL RESULTS =====")
    print(f"Best User Fitness: {max(results['user_history']):.4f}")
    print(f"Best Item Fitness: {max(results['item_history']):.4f}")

    # -------------------------
    # 5. Example recommendation demo
    # -------------------------
    print("\n===== SAMPLE RECOMMENDATIONS =====")

    users = results["users"]
    items = results["items"]

    # Get the absolute fittest user in the final population
    best_user = max(users, key=lambda u: u.fitness if u.fitness is not None else -float('inf'))
    print(f"Top 5 unique recommendations for User {best_user.user_id}:")

    # Use a dictionary to filter out the clones, keeping only the highest score for each ID
    unique_item_scores = {}
    
    for item in items:
        score = float(np.dot(best_user.vector, item.vector))
        
        if item.item_id not in unique_item_scores or score > unique_item_scores[item.item_id]:
            unique_item_scores[item.item_id] = score

    # Convert the dictionary back to a list of tuples and sort it
    sorted_scores = sorted(unique_item_scores.items(), key=lambda x: x[1], reverse=True)

    # Print the top 5 unique items
    for item_id, score in sorted_scores[:5]:
        print(f"Item {item_id} → score: {score:.4f}")

if __name__ == "__main__":
    main()