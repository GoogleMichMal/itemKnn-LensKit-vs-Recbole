import numpy as np

def _ideal_dcg(n):
    ranks = np.arange(1, n+1)
    disc = 1 / np.log2(ranks + 1)
    ideal_dcg = np.sum(disc)
    return ideal_dcg