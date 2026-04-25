from models.individuals import BaseIndividual
import numpy as np

class UserIndividual(BaseIndividual):
    def __init__(self, vector, user_id=None):
        super().__init__(vector)
        self.user_id = user_id  # Link to dataset

    def predict(self, item_vector):
        """Predict rating using dot product"""
        return float(np.dot(self.vector, item_vector))