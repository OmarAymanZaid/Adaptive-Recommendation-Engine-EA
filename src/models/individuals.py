import random
import numpy as np

class BaseIndividual:
    def __init__(self, vector):
        self.vector = np.array(vector, dtype=float)
        self.fitness = None  # Higher is better (we’ll use -MSE)

    def copy(self):
        clone = self.__class__(self.vector.copy())
        clone.fitness = self.fitness
        return clone

    def __len__(self):
        return len(self.vector)

    def __repr__(self):
        return f"{self.__class__.__name__}(fitness={self.fitness})"