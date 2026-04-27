from config.parameters import PARAMS
from evolution.evaluation import evaluate_users
from utils.seeds import set_seed
from dataloader.loader import load_dataset
from evolution.coevolution import run_coevolution


def main():

    # # -------------------------
    # # 1. Set seed (reproducibility)
    # # -------------------------
    # set_seed(PARAMS["random_seed"])

    # # -------------------------
    # # 2. Load dataset
    # # -------------------------
    # dataset = load_dataset()

    # # -------------------------
    # # 3. Run coevolution
    # # -------------------------
    # results = run_coevolution(
    #     dataset=dataset,
    #     config=PARAMS
    # )

    # # -------------------------
    # # 4. Print summary
    # # -------------------------
    # print("\n===== FINAL RESULTS =====")
    # print(f"Best User Fitness: {max(results['user_history']):.4f}")
    # print(f"Best Item Fitness: {max(results['item_history']):.4f}")

    # # -------------------------
    # # 5. Example recommendation demo
    # # -------------------------
    # print("\n===== SAMPLE RECOMMENDATIONS =====")

    # users = results["users"]
    # items = results["items"]

    # user = users[0]

    # scores = []
    # for item in items:
    #     score = user.predict(item.vector)
    #     scores.append((item.item_id, score))

    # scores.sort(key=lambda x: x[1], reverse=True)

    # for item_id, score in scores[:5]:
    #     print(f"Item {item_id} → score: {score:.4f}")
    evaluate_users(user_population, item_population, dataset)

    fits = [u.fitness for u in user_population]
    print("User fitness:", fits[:5])

if __name__ == "__main__":
    main()