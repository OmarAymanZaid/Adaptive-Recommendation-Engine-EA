# models/user.py
from models.individuals import BaseIndividual

class UserIndividual(BaseIndividual):
    def __init__(self, vector, user_id=None):
        super().__init__(vector)
        self.user_id = user_id  # Link to dataset

    def copy(self):
        # Explicitly pass the user_id to the constructor
        clone = self.__class__(self.vector.copy(), user_id=self.user_id)
        clone.fitness = self.fitness
        return clone