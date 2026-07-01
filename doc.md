# 🧬 Adaptive Recommendation Engine using Coevolutionary Algorithms

## 1. Overview

An experimental Evolutionary Algorithms (EAs) program implementing an adaptive recommendation system using coevolutionary algorithms. It provides a real-time visual dashboard to tune hyperparameters and analyze model performance on the fly.

## 2. Introduction
When a user visits a website like an online bookstore, should the system recommend every single book in the world to them, or should it be smarter and only suggest the ones they actually care about? Obviously, the second option is the way to go, and that is exactly what recommendation systems are built to do.

Recommendation systems try to predict what might interest users to provide a better experience, saving them from getting lost in a massive sea of items they have zero interest in.

There are many different types of recommendation systems and a variety of ways to implement them. One unique approach is using Evolutionary Algorithms (EAs), which is exactly the path we are exploring in this project.

## 3. High-Level View
This project is a hands-on, experimental tool built to test and visualize how Evolutionary Algorithms operate under the hood when tackling complex problems.

What This Project Is (And Isn't) <br>
It is important to clear up one thing right away: this is not a production-ready recommendation system designed to be deployed on a real website like Netflix or Amazon. Instead, it is an algorithmic playground. We are simply using a recommendation problem—figuring out user preferences for certain items—as a benchmark to see how different genetic strategies perform.

The Real-Time Dashboard <br>
The core feature of this project is its interactive GUI dashboard. Through this interface, you can dynamically tune different hyperparameters on the fly—like mutation rates, crossover rates, selection mechanics, and replacement strategies. As you adjust these settings, the dashboard updates in real time, allowing you to instantly watch how your tweaks affect the model's accuracy and how fast it learns (converges).

## 4. System Design

### 4.1 Overview <br>
At its core, our system deals with two main groups: users and items. In a normal scenario, users rate items on a scale from 1 to 5. However, they don't rate everything in the database—in fact, most users only rate a tiny fraction of items, and some might not rate anything at all.

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