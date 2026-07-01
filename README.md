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

* Each **user** is represented as a preference vector
* Each **item (book)** is represented as a feature vector
* Ratings are predicted using similarity (e.g., dot product)
* Two populations evolve together:

  * Users adapt to better represent preferences
  * Items adapt to better satisfy users

---

## ⚙️ Evolutionary Algorithm Design

### Representation

* `UserIndividual`: vector of floats
* `ItemIndividual`: vector of floats

### Populations

* User population
* Item population

### Selection Methods

* Tournament Selection
* Roulette Wheel Selection

### Crossover Operators

* One-Point Crossover
* Uniform Crossover

### Mutation Operators

* Gaussian Mutation
* Random Reset Mutation

### Survivor Selection

* Generational Replacement
* Elitism

### Termination

* Fixed number of generations

---

## 🔁 Coevolution Strategy

The system uses **cooperative coevolution**, where:

* Users are evaluated based on how accurately they predict ratings
* Items are evaluated based on how well they satisfy users

---

## 🌱 Diversity Preservation

A lightweight diversity mechanism is used to prevent premature convergence, such as:

* Fitness sharing (or)
* Controlled random re-initialisation

---

## 🧪 Experiments

The system supports experimentation with:

* Different selection methods
* Different mutation and crossover operators
* Different survivor selection strategies
* Multiple initialisation approaches

Each configuration is evaluated over multiple runs using controlled random seeds.

---

## 🖥️ Interface

A simple user interface allows:

* Running the algorithm with different parameters
* Viewing recommended books
* Observing fitness evolution

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
python src/main.py
```

### Run the interface (if applicable)

```bash
streamlit run src/ui/app.py
```

---

## 📊 Evaluation Metrics

* Mean Squared Error (MSE)
* Average fitness across generations
* Best fitness per run

---

## 🧩 Future Improvements

* Incorporate hybrid approaches
* Explore competitive coevolution variants
* Use larger real-world datasets
* Improve visualisation and interactivity

---

## 📄 License

This project is for academic purposes.
