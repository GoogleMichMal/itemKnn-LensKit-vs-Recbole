from calculate_dcg import calculate_dcg
from _ideal_dcg import _ideal_dcg

# Calculate nDCG for a user
def calculate_ndcg(top_items, test_items):
    dcg = calculate_dcg(top_items, test_items)
    ideal_dcg = _ideal_dcg(len(top_items), len(test_items))
    if ideal_dcg == 0:
        return 0  # Handle division by zero
    ndcg = dcg / ideal_dcg
    return ndcg