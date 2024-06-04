import numpy as np


"""Calculate nDCG for a user for lenskit
    
    @param n: number of ranks
    @param top_items: recommended items
    @param test_items: user items in the test set
"""
class nDCG:
    n = 0
    top_items = []
    test_items = [] 
    def __init__(self, n, top_items, test_items):
        self.n = n
        self.top_items = top_items
        self.test_items = test_items



    """Calculate ideal DCG for a user
    """
    def _ideal_dcg(self):

        # iranks:  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        iranks = np.zeros(self.n, dtype=np.float64)
        # iranks:  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        iranks[:] = np.arange(1, self.n+1)
        idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=0)
        if len(self.test_items) < self.n:
            idcg[len(self.test_items):] = idcg[len(self.test_items) - 1]

        return idcg[self.n-1]



    """Calculate DCG for a user
    """
    def calculate_dcg(self):
        dcg = 0
        for i, item in enumerate(self.top_items):
            if item in self.test_items:
                relevance = 1
            else:
                relevance = 0
            rank = i + 1
            dcg += relevance / np.log2(rank + 1)
        return dcg



    """Calculate nDCG for a user
    """
    def calculate(self):
        dcg = self.calculate_dcg()
        ideal_dcg = self._ideal_dcg()
        if ideal_dcg == 0:
            return 0  # Handle division by zero
        ndcg = dcg / ideal_dcg
        return ndcg