# Adaptive Recommendation Engine using Coevolutionary Algorithms

## 📌 Overview

This project implements a recommendation system for books using **coevolutionary algorithms**. Two interacting populations—**users** and **items (books)**—are evolved simultaneously to improve recommendation quality over time.

The system models user preferences and item characteristics as vector representations and optimises them using evolutionary algorithms.

---

## 🎯 Objectives

* Formulate the recommendation problem as an **optimisation task**
* Implement **cooperative coevolution** between users and items
* Evaluate recommendation quality using fitness functions
* Experiment with different evolutionary strategies
* Provide a simple interface to visualise recommendations and parameter effects

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
├── data/
├── src/
│   ├── main.py
│   ├── config/
│   ├── data/
│   ├── models/
│   ├── evolution/
│   ├── utils/
│   ├── experiments/
│   └── ui/
│
├── results/
└── report/
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

## 👥 Authors

* Omar Ayman
* Team Members

---

## 📄 License

This project is for academic purposes.
