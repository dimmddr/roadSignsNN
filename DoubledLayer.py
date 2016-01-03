import numpy as np


class DoubledLayer:
    def __init__(self, filters_count=10, filterW=5, filterH=5, seed=16):
        np.random.seed(seed)
        self.filters = np.random.uniform(size=(filters_count, filterW, filterH))

    def detWeights(self):
        return self.filters
