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