# 🧬 Adaptive Recommendation Engine using Coevolutionary Algorithms

## 1. Overview

An experimental Evolutionary Algorithms (EAs) program implementing an adaptive recommendation system using coevolutionary algorithms. It provides a real-time visual dashboard to tune hyperparameters and analyze model performance on the fly.

## 2. Introduction
When a user visits a website like an online bookstore, should the system recommend every single book in the world to them, or should it be smarter and only suggest the ones they actually care about? Obviously, the second option is the way to go, and that is exactly what recommendation systems are built to do.

Recommendation systems try to predict what might interest users to provide a better experience, saving them from getting lost in a massive sea of items they have zero interest in.

There are many different types of recommendation systems and a variety of ways to implement them. One unique approach is using Evolutionary Algorithms (EAs), which is exactly the path we are exploring in this project.

## 3. High-Level View
This project is a hands-on, experimental tool built to test and visualize how Evolutionary Algorithms operate under the hood when tackling complex problems.

**What This Project Is (And Isn't)** <br>
It is important to clear up one thing right away: this is not a production-ready recommendation system designed to be deployed on a real website like Netflix or Amazon. Instead, it is an algorithmic playground. We are simply using a recommendation problem—figuring out user preferences for certain items—as a benchmark to see how different genetic strategies perform.

**The Real-Time Dashboard** <br>
The core feature of this project is its interactive GUI dashboard. Through this interface, you can dynamically tune different hyperparameters on the fly—like mutation rates, crossover rates, selection mechanics, and replacement strategies. As you adjust these settings, the dashboard updates in real time, allowing you to instantly watch how your tweaks affect the model's accuracy and how fast it learns (converges).

## 4. System Design

### 4.1 Overview <br>
At its core, our system deals with two main groups: users and items. In a normal scenario, users rate items on a scale from 1 to 5. However, they don't rate everything in the data—in fact, most users only rate a tiny fraction of items, and some might not rate anything at all.

Our goal is to predict the missingratings. If we can look at a user's past behavior (the items they liked and gave high ratings to, or disliked and gave low ratings to), we can guess how they would rate an item they haven't seen yet. Instead of using traditional machine learning gradient descent to solve this, we are using Evolutionary Algorithms (EAs).

**How We Represent the Data** <br>
We represent every user as a preference vector and every item as a feature vector. We then create two distinct populations:

1. A population of all users in the system.
2. A population of all items in the system.

**The Evolution Loop** <br>
We evolve these two populations separately using an interleaved approach. When we focus on the user population (and the exact same logic applies when it's the items' turn), the process goes like this:

- Initialization: We start with random vectors for all the users.

- Evaluation (Fitness Calculation): To find out how good a user's vector is, we look at all the items that specific user has actually rated in our dataset. We pull those item vectors from the item population and calculate the dot product between the user vector and each item vector. This dot product represents our predicted score.

- Error Calculation: We compare our prediction to the actual rating the user gave and calculate the error. We accumulate these errors across all the items the user rated to get the Mean Squared Error (MSE). Because genetic algorithms try to maximize fitness, we use the negative MSE (-MSE) as the fitness score.

- Reproduction: We run standard genetic algorithm operations—like selection, crossover, and mutation—to create fitter individuals and optimize these vectors over multiple generations until the error stops dropping (converges).

Once we finish a cycle with the users, we switch and do the exact same thing for the items population. By alternating back and forth, we eventually end up with highly optimized vectors for both users and items.

**Making Recommendations** <br>
Once the evolution process is complete and the vectors are trained, the system can generalize to unseen data. The user interface takes these final, evolved vectors and runs a full prediction pass. For any selected user, it calculates the dot product against every single item in the database—regardless of whether the user originally rated it or not—allowing us to rank and recommend the absolute best matches.

### 4.2 Dataset

Instead of using a real-world dataset (like MovieLens or Goodreads), this project uses a synthesized dataset. This allows us to control the exact environment, introduce specific amounts of noise, and set up a clear "ground truth" ruleset to see if our Evolutionary Algorithm can successfully reverse-engineer it.

**Dataset Structure** <br>
The dataset is structured using two Python dictionaries to make lookups fast and straightforward:

- **User Dictionary (user_dict):** The keys are unique user IDs, and the values are lists of tuples. Each tuple contains an item ID and the rating that the user gave to that item:
{user_id: [(item_id_1, rating), (item_id_2, rating), ...]}

- **Item Dictionary (item_dict):** The keys are unique item IDs, and the values are lists of tuples. Each tuple contains a user ID and the rating that user gave to the item:
{item_id: [(user_id_1, rating), (user_id_2, rating), ...]}

**How the Synthetic Data is Generated** <br>
To create this dataset, the system follows a step-by-step mathematical pipeline:

1. Create True Latent Vectors: We initialize separate lists of "true" latent vectors for both users and items using a bounded uniform distribution ($\mathcal{U}(-1, 1)$). These represent the hidden, abstract features of our data (e.g., a user's preference for a genre vs. an item containing that genre).

2. Calculate the Dot Product: In a nested loop, we compute the inner product between every single user vector and item vector. The dot product measures alignment; if the vectors point in similar directions in our latent space, it yields a high positive score (strong alignment), and if they oppose each other, it yields a negative score (poor alignment).

3. Normalize by Dimension: Because adding up multiple dimensions could cause the total product to scale wildly (e.g., between -10 and 10 for a 10-dimension vector), we divide the result by the vector dimension (dim). This brings the score back down into a manageable $[-1, 1]$ neighborhood.

4. Inject Stochastic Realism (Noise): Real human behavior isn't a perfect mathematical equation; a user might love a book but rate it poorly just because they had a bad day. We inject Gaussian noise ($\epsilon \sim \mathcal{N}(0, \text{noise})$) to break the "perfect" math. This makes the dataset messy and realistic, ensuring our evolutionary algorithm is robust enough to handle real-world imperfections instead of just overfitting to clean numbers.

5. Scale to Target Range: Finally, we use a scaling formula to transform the normalized, noisy value into a traditional $[1, 5]$ star-rating scale, applying a safety guard (np.clip) to handle any extreme noisy outliers.

**Simulating Sparsity** <br>
In real life, users only rate a tiny fraction of available items. To mirror this, we introduce a sparsity condition during data generation. If a randomly generated probability is less than our target sparsity threshold, we skip saving that interaction entirely:

```
# Sparsity condition (simulate missing ratings)
if np.random.rand() < sparsity:
    continue
```

**Academic & Theoretical Validation** <br>
While creating a simulated dataset might feel arbitrary at first glance, this implementation actually follows the standard academic blueprint for Synthetic Data Generation in Matrix Factorization (MF) and Collaborative Filtering.

When evaluating new or alternative optimization methods (like EAs) against machine learning problems, using simulated datasets with controlled Gaussian noise and sparsity loops is the textbook method to benchmark convergence rates, parameter sensitivity, and resilience to overfitting.

If you need to cite the literature or theory behind these design choices, they are well-supported by these foundational texts:

1. The Core Concept (Latent Vectors & Dot Product):

    - Koren, Y., Bell, R., & Volinsky, C. (2009). "Matrix Factorization Techniques for Recommender Systems." IEEE Computer.

    - Why it matches our code: This is the seminal paper on the topic. It outlines modeling users and items as vectors in a joint latent factor space where interactions are captured by their inner product.

2. The Evaluation Framework (Synthetic Generation & Noise):

    - Aggarwal, C. C. (2016). "Recommender Systems: The Textbook." Springer.

    - Why it matches our code: Chapter 3 covers collaborative filtering and matrix factorization, detailing how simulated datasets are frequently used to evaluate algorithm behavior under controlled conditions.

### 4.3 Representation

In almost any recommendation system that relies on making predictions (whether it uses traditional machine learning or evolutionary algorithms), users and items must be represented mathematically. We do this by converting them into vectors—which are essentially just lists of numbers.

**User Vectors vs. Item Vectors** <br>
- **The User Vector:** This represents a user's personal tastes or preferences. For example, if our system recommends books, the numbers in the user's vector show how much they like specific attributes, such as {action, classic literature, textbook}.

- **The Item Vector:** This represents the actual characteristics present in a specific item. For a book, its vector numbers reflect how much of those same traits it possesses, like {amount of action, classic style, academic textbook depth}.

The goal is to see how well these two vectors align. The closer a user’s preference numbers match an item’s feature numbers, the more suitable that item is for the user. 

Mathematically, we measure this suitability using the dot product, which multiplies the two vectors together to give us a single alignment score.

**Why We Call Them "Latent" Vectors** <br>
In recommendation systems textbooks, we will often see these vectors referred to as latent vectors (and the space they live in is called the latent factor space). The word latent simply means hidden or concealed.

In a perfect world, we might manually tag every book with explicit features (like "20% romance, 80% sci-fi") and survey every user on exactly what they want. But in reality, human tastes are messy, and tagging millions of items by hand is impossible. By using latent vectors, we don't actually hardcode what each index in the vector stands for. We don't say "index 0 is action and index 1 is page count." Instead, we just give the algorithm a blank set of numbers (dimensions) for each user and item. As the Evolutionary Algorithm runs and tries to minimize the rating error, it automatically figures out what these hidden factors should be.The algorithm discovers underlying patterns on its own—such as grouping books by unspoken sub-genres or writing styles—and maps them into a shared mathematical "taste space" without us ever having to explicitly define them.

### 4.4 Models and Classes
To translate our evolutionary approach into clean, reusable Python code, we use an Object-Oriented Design. Each chromosome (or entity vector) in our populations is represented as an individual object.

**Class Hierarchy Overview** <br>
Instead of writing separate, redundant logic for users and items, we implement a base individual class that handles all common mathematical and genetic operators. The specific user and item classes then inherit from this base class.

```
          +-------------------------+
          |     BaseIndividual      |  <--  Handles vector math, cloning,
          +-------------------------+       and tracking fitness (-MSE)
                       |
         +-------------+-------------+
         |                           |
         v                           v
+------------------+       +------------------+
|  UserIndividual  |       |  ItemIndividual  |  <--- Adds unique database IDs
+------------------+       +------------------+       (user_id / item_id)

```
**Core Classes Breakdown** <br>

```1. BaseIndividual```
<br> This is the parent class for any individual element in our genetic population. It encapsulates the core properties required by the Evolutionary Algorithm.

- vector: A NumPy array (np.array) representing the chromosome's real-valued genes (the latent factors).

- fitness: Tracks how well this individual describes the data. We initialize this to None, and it is later filled with the negative Mean Squared Error (-MSE).

- predict(other_vector): The core prediction function. It calculates the raw dot product ($\text{User} \cdot \text{Item}$) between this individual's vector and a target vector to approximate a rating score.

- copy(): Creates a deep copy (clone) of the individual, preserving its current vector values and fitness score. This is crucial during selection and crossover so that modifying an offspring does not accidentally ruin the parent.

<br>

```2. UserIndividual```
<br>Inherits directly from BaseIndividual. It represents a single user within the user population.

- user_id: An explicit identifier that links this specific individual vector back to its unique profile entry in our user_dict dataset.

<br>

```3. ItemIndividual```
<br>Inherits directly from BaseIndividual. It represents a single item within the item population.

- item_id: An explicit identifier that links this specific individual vector back to its unique profile entry in our item_dict dataset.

### 4.5 Functions and Modules

To make the codebase modular and easy to manage, the system is broken down into specific folders (modules), where each file has a focused job. Below is the breakdown of the primary modules and their principal functions.

**1. Data Loader (```dataloader/loader.py```)** <br>
This module handles setting up the dataset that the algorithm will train on.

- load_dataset(): The main entry point that chooses whether to load real data or build a simulated environment.

- generate_synthetic_data(): Handles the Low-Rank Matrix Factorization math loop to build a "ground truth" dataset from scratch.

- load_real_dataset(): Handles reading external, real-world data files.

**2. Evaluation Utilities (```utils/metrics.py & utils/seeds.py```)** <br>
Helper files to keep calculations clean and ensure our experiments are reproducible.

- get_predictions_and_truths(): Compares the algorithm's predicted ratings against the true target scores.

- calculate_rmse() / calculate_mae(): Standard error metrics to calculate model accuracy.

- set_seed(): Locks random number generators (numpy and random) so running the same experiment twice yields the exact same results.

**3. Population Management (```evolution/population.py```)** <br>
Responsible for spinning up the collections of chromosomes at the very beginning of the program.

- initialize_populations(): The top-level function that triggers the creation of both the user and item populations.

- create_user_population() / create_item_population(): Generates individual vectors using either a flat random uniform setup (random_uniform_vector) or standard normal curves (gaussian_vector).

**4. Genetic Operators (mutation.py, crossover.py, selection.py)** <br>
These modules house the classic textbook evolutionary mechanisms.

- mutate(): Alters genes randomly. Supports both resetting individual traits entirely (random_reset_mutation) or adding small nudges (gaussian_mutation).

- crossover(): Combines genetic data from two parents. Supports breaking the vector at a single point (one_point_crossover) or swapping genes randomly element-by-element (uniform_crossover).

- select_parents(): Gathers highly fit individuals into a mating pool using either Tournament Selection or Roulette Wheel Selection.

**5. Replacement Strategies (evolution/replacement.py)** <br>
Determines how the system transitions from an old generation to a newly born generation of offspring.

- generational_replacement(): Replaces the entire parent population with the new offspring.

- elitist_replacement(): Keeps a small percentage of the top-performing historical parents no matter what.

- species_preserving_replacement(): Our specialized mechanism that groups individuals by their immutable ID to make sure high-performing user or item profiles are never accidentally lost to destructive mutations.

**6. The Execution Core (evaluate.py & coevolution.py)** <br>
Where everything comes together to actually run the training loop.

- evaluate_users() / evaluate_items(): Loops through the respective populations and calculates individual fitnesses (using -MSE) by comparing vector dot products (predict_rating) against active dataset records.

- run_coevolution(): The main evolutionary driver. It coordinates the step-by-step sequential execution: pausing the items to evolve the users, then pausing the users to evolve the items.

## 5. Evolutionary Algorithm Design

This chapter dives into the exact mechanics of our genetic algorithm, explaining how we adapted textbook evolutionary concepts to fit the unique requirements of a Matrix Factorization problem.

### 5.1 Population Mapping and the Nature of the Problem

In a Genetic Algorithm textbook, a population is a collection of candidate solutions to the entire problem. For example, if you are solving the Traveling Salesperson Problem (TSP), a single chromosome represents a complete, finished route, and your population might consist of 100 different route variations competing against each other.

However, in a recommendation system based on Matrix Factorization, a "complete solution" is the entire massive matrix containing every single user and item. If a single chromosome had to represent the entire universe of users and items at once, the chromosome size would be gigantic, making the search space incredibly inefficient to explore globally.

To solve this, our system treats individual components as the chromosomes:

- Each chromosome represents a single, specific user vector (e.g., User #5) or a single, specific item vector (e.g., Item #12).

- Therefore, the population size is not an arbitrary number (like 100 or 500); it is the exact number of unique users ($M$) or items ($N$) currently in our dataset.

**Is This a Limitation?** <br>
Yes and no. This setup forces a strict 1:1 mapping between a physical entity ID and a chromosome. The downside is that you don't have multiple variations of "User #5" competing in the same population pool at the same time. Instead, those variations are explored across time (generations) rather than space (simultaneous random solutions).

### 5.2 Chromosome Representation and Genetic Operators
Because our latent factors are continuous, decimal-based preferences, we use a real-valued representation for our chromosomes. This choice dictates the specific types of mutation and crossover operators we can use.

**1. Mutation**
<br>Instead of flipping binary bits, our mutation operators must modify continuous floating-point numbers. We implemented two modes:

- Uniform Mutation: Replaces a chosen gene with a totally new random value within our bounded uniform range.

- Non-Uniform (Gaussian) Mutation: Nudges the existing gene value by adding a small amount of random noise from a Gaussian distribution. This allows for fine-tuning vectors as the generations progress.

**2. Crossover**
<br>
Crossover blends the characteristics of two parent vectors to create new offspring profiles:

- One-Point Crossover: Chooses a random split point along the vector dimension; genes before the split come from Parent A, and genes after come from Parent B.

- Uniform Crossover: Loops through every vector index individually, tossing a coin to decide whether that specific trait should be inherited from Parent A or Parent B.

### 5.3 Selection and Mating Mechanics
Our system breaks selection down into two distinct phases: choosing who enters the general mating pool, and choosing the exact pairs that reproduce.

**1. Population Selection (The Mating Pool)**
<br>
To fill our mating pool with high-performing individuals, we support two classic textbook strategies:

- Tournament Selection: Small groups are chosen at random, and the individual with the best fitness wins a spot in the pool.

- Roulette Wheel Selection: Individuals are assigned a slice of a selection wheel proportional to their fitness, giving better vectors a higher probability of being chosen.

**2. Parent Pairing**
<br>
Once the highly fit individuals are chosen and placed into the mating pool array (item_selected or user_selected), we use a lightweight sequential coupling mechanism to pair them up. The code simply loops through adjacent elements:

```
p1 = item_selected[i]
p2 = item_selected[(i + 1) % len(item_selected)]
```

### 5.4 Management Strategy and Fitness Evaluation

**1. Management Model**
<br>
The overarching system runs on a generational model where an entire generation of users or items is evaluated, selected, and bred to create a new generation of offspring.

**2. Evaluation & Fitness Function**
<br>
As established in our system design, the algorithm evaluates vectors by predicting known interaction scores using the dot product. Because genetic algorithms naturally look for the highest possible value, our fitness function is the negative Mean Squared Error (-MSE). Minimizing the rating error maximizes this fitness value toward zero.

### 5.5 Custom Replacement and Preserving Diversity
Because our population maps 1:1 to explicit dataset IDs, standard textbook replacement strategies would completely break our system. If we used a pure generational replacement, a bad mutation could completely wipe out the best-known vector for User #5, causing the system to completely forget that user's optimal profile.

To prevent this, we implemented a highly specialized **Species-Preserving Replacement Strategy**.

**Clarifying the Taxonomy: Is it Formal Speciation?**
<br>
- Is it formal Speciation or Niching? No. In classic evolutionary algorithms (like NEAT), "species" are dynamically clustered based on genetic distance (grouping similar-looking vectors into a niche). Our code does not calculate distance metrics between users to group them.

- What is it instead? ID-based Survival. Our code treats each individual unique ID slot as its own immutable "species slot."

**The Mechanism**
<br>
Structurally, this strategy is equivalent to a localized Steady-State $(\mu + \lambda)$ survival per slot, scaled across the entire population.The algorithm pools the parents ($\mu$) and the newly generated offspring ($\lambda$), groups them by their permanent database identities (user_id or item_id), and preserves exactly one elite vector per ID.

**Preserving Diversity**
<br>
This strict identity-based elitism acts as our primary shield for preserving diversity and fighting Genetic Drift. It guarantees that the best-found vector representation for a specific user or item can never accidentally go extinct due to destructive mutations or crossover steps, keeping our overall "taste space" stable and robust.

## 6. Summary

This project successfully builds an experimental, interactive playground designed to explore the mechanics of Evolutionary Algorithms (EAs) within a recommendation system context.

By utilizing a Low-Rank Matrix Factorization approach, we shifted away from traditional machine learning methods like gradient descent, opting instead to optimize user and item latent vectors through genetic operations. Rather than treating a single chromosome as a global solution, the system introduces a 1:1 entity-to-chromosome mapping, scaling the population sizes exactly to the number of unique users and items in the system.

To handle the mutual dependency of user preferences and item features, the system utilizes Cooperative Coevolution (CCE), optimizing each population in an alternating, step-by-step cycle. Furthermore, to combat genetic drift and prevent destructive mutations from erasing highly fit profiles, a specialized Species-Preserving Replacement Strategy was implemented, treating each unique database ID as an immutable slot that guarantees localized survival.

Paired with a real-time GUI dashboard, the program functions as a highly visual simulation tool. It allows users to dynamically alter genetic operators—such as mutation rates, crossover styles, and selection methods—and instantly observe their direct impact on model convergence and Mean Squared Error (-MSE). Ultimately, the project demonstrates that evolutionary frameworks, when tailored with specialized replacement and coevolutionary strategies, can effectively navigate and optimize highly complex, sparse, multi-variable environments.
