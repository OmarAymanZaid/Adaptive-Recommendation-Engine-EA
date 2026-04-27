# models/item.py
from models.individuals import BaseIndividual

class ItemIndividual(BaseIndividual):
    def __init__(self, vector, item_id=None):
        super().__init__(vector)
        self.item_id = item_id

    def copy(self):
        # Explicitly pass the item_id to the constructor
        clone = self.__class__(self.vector.copy(), item_id=self.item_id)
        clone.fitness = self.fitness
        return clone