def make_implicit_movielens(data):
    # set ratings smaller or equal to three to zero
    data["rating"][data["rating"] <= 3] = 0
    # set ratings higher than three to one
    data["rating"][data["rating"] > 3] = 1
    # keep only non-zero ratings
    data = data[data["rating"] == 1]
    return data
