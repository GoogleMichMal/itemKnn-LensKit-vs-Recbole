import numpy as np

# Calculate DCG for a user
# top_items: recommended items
# test_items: user items in the test set
def calculate_dcg(top_items, test_items):

    dcg = 0
    for i, item in enumerate(top_items):
        if item in test_items:
            relevance = 1
        else:
            relevance = 0
        rank = i + 1
        dcg += relevance / np.log2(rank + 1)
    return dcg
