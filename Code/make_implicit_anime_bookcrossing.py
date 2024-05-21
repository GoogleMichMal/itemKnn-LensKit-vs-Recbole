def make_implicit_anime_bookcrossing(data):
    # set ratings smaller or equal to three to zero
    data["rating"][data["rating"] <= 5] = 0
    # set ratings higher than three to one
    data["rating"][data["rating"] > 5] = 1
    # keep only non-zero ratings
    data = data[data["rating"] == 1]
    return data

