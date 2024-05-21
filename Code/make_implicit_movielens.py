import pandas as pd

ml100k = pd.read_csv("Data/ml-100k/u.data", header=None, sep="\t", names=["user", "item", "rating", "timestamp"])
print(ml100k)

def make_implicit_movielens(data):
    # set ratings smaller or equal to three to zero
    data["rating"][data["rating"] <= 3] = 0
    # set ratings higher than three to one
    data["rating"][data["rating"] > 3] = 1
    # keep only non-zero ratings
    data = data[data["rating"] == 1]
    return data
