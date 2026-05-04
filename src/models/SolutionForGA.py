#New class to represent a solution in the genetic algorithm

import numpy as np
from models.individuals import BaseIndividual

class SolutionIndividual(BaseIndividual):
    def __init__(self, vector, user_ids, item_ids, vector_dim):
        super().__init__(vector)
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.vector_dim = vector_dim
        
        # Precompute index mappings for fast O(1) lookup during evaluation
        self._user_idx_map = {uid: i for i, uid in enumerate(user_ids)}
        self._item_idx_map = {iid: i for i, iid in enumerate(item_ids)}
        
        # Determine exactly where in the giant flat array the item vectors start
        self._user_offset = 0
        self._item_offset = len(user_ids) * vector_dim

    def copy(self):
        # Create a deep clone, carrying over the fitness score
        clone = self.__class__(
            self.vector.copy(), 
            self.user_ids, 
            self.item_ids, 
            self.vector_dim
        )
        clone.fitness = self.fitness
        return clone
    
    def get_user_vector(self, user_id):
        idx = self._user_idx_map.get(user_id)
        if idx is None: return None
        start = self._user_offset + (idx * self.vector_dim)
        return self.vector[start : start + self.vector_dim]

    def get_item_vector(self, item_id):
        idx = self._item_idx_map.get(item_id)
        if idx is None: return None
        start = self._item_offset + (idx * self.vector_dim)
        return self.vector[start : start + self.vector_dim]