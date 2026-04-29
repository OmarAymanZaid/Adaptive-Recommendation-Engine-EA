import sys
import os
# Force Python to recognize the 'src' folder as the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np

# Import your finished backend modules
from config.parameters import PARAMS
from utils.seeds import set_seed
from dataloader.loader import load_dataset
from evolution.coevolution import run_coevolution

def run_engine_gui():
    """
    This function triggers when the 'Run Evolution' button is clicked.
    It reads the UI inputs, runs the backend, and prints the results to the window.
    """
    # 1. Update the PARAMS dictionary with the numbers typed into the UI
    try:
        PARAMS["mutation_rate"] = float(mut_rate_var.get())
        PARAMS["tournament_size"] = int(tourn_size_var.get())
    except ValueError:
        output_box.insert(tk.END, "Error: Please enter valid numbers for parameters.\n")
        return
        
    # 2. Clear the text box and show a loading message
    output_box.delete(1.0, tk.END)
    output_box.insert(tk.END, "Running Evolution (100 Generations)...\nPlease wait, this may take a few seconds.\n\n")
    app.update() # Force the UI to refresh before the engine freezes it
    
    # 3. Run the exact same engine from main.py
    set_seed(PARAMS["random_seed"])
    dataset = load_dataset()
    results = run_coevolution(dataset=dataset, config=PARAMS)
    
    # 4. Extract results safely
    users = results["users"]
    items = results["items"]
    best_user = max(users, key=lambda u: u.fitness if u.fitness is not None else -float('inf'))
    
    # 5. Display the final fitness scores
    output_box.insert(tk.END, f"===== FINAL RESULTS =====\n")
    output_box.insert(tk.END, f"Best User Fitness: {max(results['user_history']):.4f}\n")
    output_box.insert(tk.END, f"Best Item Fitness: {max(results['item_history']):.4f}\n\n")
    output_box.insert(tk.END, f"Top 5 unique recommendations for User {best_user.user_id}:\n")
    
    # 6. Deduplicate and display the top 5 items
    unique_item_scores = {}
    for item in items:
        score = float(np.dot(best_user.vector, item.vector))
        if item.item_id not in unique_item_scores or score > unique_item_scores[item.item_id]:
            unique_item_scores[item.item_id] = score
            
    sorted_scores = sorted(unique_item_scores.items(), key=lambda x: x[1], reverse=True)
    
    for item_id, score in sorted_scores[:5]:
        output_box.insert(tk.END, f"Item {item_id} → score: {score:.4f}\n")

# ==========================================
# Build the Desktop Window
# ==========================================
app = tk.Tk()
app.title("Adaptive Recommendation Engine")
app.geometry("520x450")

# Create a clean layout frame
controls = ttk.LabelFrame(app, text="Engine Parameters")
controls.pack(padx=15, pady=15, fill="x")

# Mutation Rate UI
ttk.Label(controls, text="Mutation Rate:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
mut_rate_var = tk.StringVar(value=str(PARAMS["mutation_rate"]))
ttk.Entry(controls, textvariable=mut_rate_var, width=10).grid(row=0, column=1, padx=10, pady=10)

# Tournament Size UI
ttk.Label(controls, text="Tournament Size:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
tourn_size_var = tk.StringVar(value=str(PARAMS["tournament_size"]))
ttk.Entry(controls, textvariable=tourn_size_var, width=10).grid(row=1, column=1, padx=10, pady=10)

# Run Button
run_btn = ttk.Button(controls, text="Run Evolution Engine", command=run_engine_gui)
run_btn.grid(row=0, column=2, rowspan=2, padx=30)

# Output Display Box
output_box = scrolledtext.ScrolledText(app, width=55, height=15, font=("Consolas", 10))
output_box.pack(padx=15, pady=5)

# Start the application
if __name__ == "__main__":
    app.mainloop()