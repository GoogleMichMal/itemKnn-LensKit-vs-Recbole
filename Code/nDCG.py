import numpy as np
from recbole.evaluator.base_metric import TopkMetric

class nDCG_LK:
    """Calculate nDCG for a user for lenskit

    Args:
        n (int): number of ranks
        top_items (array): recommended items
        test_items (array): user items in the test set
    """
    n = 0
    top_items = []
    test_items = [] 
    def __init__(self, n, top_items, test_items):
        self.n = n
        self.top_items = top_items
        self.test_items = test_items



    def _ideal_dcg(self):
        """Calculate ideal DCG for a user
        """
        # iranks:  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        iranks = np.zeros(self.n, dtype=np.float64)
        # iranks:  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        iranks[:] = np.arange(1, self.n+1)
        idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=0)
        if len(self.test_items) < self.n:
            idcg[len(self.test_items):] = idcg[len(self.test_items) - 1]

        return idcg[self.n-1]



    def calculate_dcg(self):
        """Calculate DCG for a user
        """
        dcg = 0
        for i, item in enumerate(self.top_items):
            if item in self.test_items:
                relevance = 1
            else:
                relevance = 0
            rank = i + 1
            dcg += relevance / np.log2(rank + 1)
        return dcg



    
    def calculate(self):
        """Calculate nDCG for a user

        Returns:
            float: nDCG@n value for the user
        """
        dcg = self.calculate_dcg()
        ideal_dcg = self._ideal_dcg()
        if ideal_dcg == 0:
            return 0  # Handle division by zero
        ndcg = dcg / ideal_dcg
        return ndcg
    



class nDCG_RB(TopkMetric):
    """nDCG implementation of RecBole (not self-made)
    
    
    We just copied it in order to make sure that the nDCG calculation is consistent between the two frameworks. Furthermore
    we commented the calculation, to make it easier to understand the code.

    Credits: https://github.com/RUCAIBox/RecBole/blob/master/recbole/evaluator/metrics.py
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate(self, dataobject):
        # pos_index: a bool matrix, shape user * k. The item with the (j+1)-th highest score of i-th user is pos if pos_index[i][j] is True
        # pos_len: a vector representing the number of positive items for each user
        pos_index, pos_len = self.used_info(dataobject)
        result = self.metric_info(pos_index, pos_len)
        metric_dict = self.topk_result("ndcg", result)
        return metric_dict
    
    # recbole-ndcg implementation
    def metric_info(self, pos_index, pos_len):
        # len_rank: [10, 10, 10, 10, ... userNumber]
        len_rank = np.full_like(pos_len, pos_index.shape[1])
        idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)
        
        # iranks:  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ...]
        iranks = np.zeros_like(pos_index, dtype=np.float)
        # iranks:  [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ...]
        iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
        idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
        for row, idx in enumerate(idcg_len):
            # for user with less than 10 positive items, fill the rest with the last valid value
            idcg[row, idx:] = idcg[row, idx - 1]

        # ranks:  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ...]
        ranks = np.zeros_like(pos_index, dtype=np.float)
        # ranks:  [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ...]
        ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)

        dcg = 1.0 / np.log2(ranks + 1)
        dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)
        result = dcg / idcg
        return result