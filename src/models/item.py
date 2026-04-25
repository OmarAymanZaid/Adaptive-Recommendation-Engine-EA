from models.individuals import BaseIndividual
import numpy as np

class ItemIndividual(BaseIndividual):
    def __init__(self, vector, item_id=None):
        super().__init__(vector)
        self.item_id = item_id