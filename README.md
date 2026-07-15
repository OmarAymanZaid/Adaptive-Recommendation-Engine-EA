# Adaptive Recommendation Engine using Coevolutionary Algorithms

## 📌 Overview

This project is an experimental evaluation tool built to test, analyze, and visualize how Evolutionary Algorithms (EAs) operate under the hood when tackling complex, multi-variable problems.

Instead of traditional gradient-descent machine learning, this system utilizes Cooperative Coevolution (CCE) to optimize vector representations in a simulated Matrix Factorization environment. It features a real-time interactive dashboard to tune hyperparameters dynamically and instantly observe how those adjustments impact model performance and convergence.

⚠️ Note: This is a benchmark playground and simulation tool for testing genetic strategies, not a production-ready recommender system designed for deployment on real websites

---

## 🎯 Objectives

* Formulate the recommendation matrix factorization problem as an evolutionary optimization task.
* Implement cooperative coevolution between mutually dependent user and item populations.
* Evaluate vector representations effectively using negative Mean Squared Error (-MSE) as fitness.
* Provide an interactive, real-time interface to visualize hyperparameters, convergence rates, and final recommendations.

---

## 🧠 Core Idea

The system models collaborative filtering using real-valued latent factor vectors:

* **User Population:** A collection of preference vectors mapping 1:1 to unique user IDs **(M)**.
* **Item Population:** A collection of feature vectors mapping 1:1 to unique item IDs **(N)**.
* **Interaction Prediction:** Ratings are approximated by calculating the mathematical alignment (inner dot product) between a user vector and an item vector ($\text{User} \cdot \text{Item}$).

**The Synthetic Data Loader**
<br>
To test the limits of the algorithm, the system generates benchmark datasets via a Low-Rank Matrix Factorization architecture. It establishes a strict "ground truth" ruleset by multiplying uniform latent vectors ($\mathcal{U}(-1, 1)$), normalizing by vector dimension, injecting Gaussian noise ($\epsilon \sim \mathcal{N}(0, \text{noise})$) for human randomness, and downsampling to a targeted missingness level using a stochastic sparsity filter.

---

## ⚙️ Evolutionary Algorithm Design

### Representation

* `UserIndividual`: Real-valued vector of floats bound to a specific `user_id`.
* `ItemIndividual`: Real-valued vector of floats bound to a specific `item_id`.

### Genetic Operators

* **Selection Methods:** Tournament Selection and Roulette Wheel Selection (to fill the mating pool). Parents are coupled via a lightweight sequential pairing loop.

* **Crossover Operators:** One-Point Crossover and Uniform Crossover.

* **Mutation Operators:** Gaussian Mutation (adding small random nudges) and Random Reset Mutation (replacing a gene entirely).

### Management & Replacement Strategies

* Generational Model execution.

* **Replacement Modes:** Generational Replacement, Elitist Replacement, and a specialized Species-Preserving Replacement Strategy.

### Termination

* Fixed number of generations

---

## 🔁 Coevolution Strategy

Because the fitness landscapes of users and items are mutually dependent, they cannot optimize globally in isolation. The system implements an interleaved sequential execution mimicking machine learning concepts like Alternating Least Squares (ALS):

1. Freeze the Item Population: Evolve user vectors to better predict existing dataset ratings.

2. Freeze the User Population: Evolve item vectors to better satisfy user preference vectors.

---

## 🌱 Diversity Preservation & Anti-Drift

Because the population size maps 1:1 to explicit physical IDs, standard replacement risks completely wiping out a user's best-known profile due to a destructive mutation. To counter this, the custom Species-Preserving Replacement Strategy acts as our primary shield against Genetic Drift. It treats each unique ID as an immutable "species slot," pools parents ($\mu$) and offspring ($\lambda$), and executes a localized Steady-State $(\mu + \lambda)$ survival policy that preserves exactly one elite vector per ID.

---

## 🧪 Experiments

The system supports experimentation with:

* Different selection methods
* Different mutation and crossover operators
* Different survivor selection strategies
* Multiple initialisation approaches

Each configuration is evaluated over multiple runs using controlled random seeds.

---

## 🖥️ Recommendation Dashboard & Interface

The GUI dashboard acts as a visual simulator allowing you to:

* Run the algorithm and configure parameters (mutation/crossover rates, population initializers, selection types, and replacement taxonomies) on the fly.

* View live training statistics, global Mean Squared Error (MSE), and real-time fitness progression.

* Select a user and display their Top-$N$ best-fitting recommendations calculated by scoring the evolved vector against all items.

* View a relative strength visual progress bar (██████░░░░) that color-shifts based on alignment metrics (e.g., green/blue for positive alignment, red for opposing/negative alignment).

---

## 📂 Project Structure

```
adaptive-recommendation-ea/
│
├── README.md
├── requirements.txt
├── data/
└── src/
    ├── config/
    │   └── parameters.py
    │
    ├── dataloader/
    │   └── loader.py
    │
    ├── evolution/
    │   ├── coevolution.py
    │   ├── crossover.py
    │   ├── evaluation.py
    │   ├── mutation.py
    │   ├── population.py
    │   ├── replacement.py
    │   ├── run_standard_ga.py
    │   └── selection.py
    │
    ├── experiments/
    │   ├── runner_2.py
    │   └── runner.py
    │
    ├── models/
    │   ├── individuals.py
    │   ├── item.py
    │   ├── SolutionForGA.py
    │   └── user.py
    │
    ├── ui/
    │   └── gui.py
    │
    └── utils/
        ├── metrics.py
        └── seeds.py


```

---

## 🚀 Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Run the system

```bash
python -m src.ui.app
```

---

## 📊 Evaluation Metrics

* Negative Mean Squared Error (-MSE): Primary fitness function guiding chromosomes into a shared taste space.

* Root Mean Squared Error (RMSE) & Mean Absolute Error (MAE): Standard accuracy benchmarks calculated over known interactions.

* Convergence Speed: Evaluated by observing average and peak population fitness across generations via the GUI.

---

## 🧩 Future Improvements

* Incorporate hybrid approaches
* Explore competitive coevolution variants
* Use larger real-world datasets
* Improve visualisation and interactivity

---

## 📄 License

This project is for academic purposes.
