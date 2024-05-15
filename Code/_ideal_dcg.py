import numpy as np

def _ideal_dcg(n, test_items):

    # iranks:  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    iranks = np.zeros(n, dtype=np.float64)
    # iranks:  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    iranks[:] = np.arange(1, n+1)
    idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=0)
    if test_items < n:
        idcg[test_items:] = idcg[test_items - 1]

    return idcg[n-1]
