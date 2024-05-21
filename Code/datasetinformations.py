import pandas as pd

def ml100k_information():
    ml100k = pd.read_csv("Data/ml-100k/u.data", sep="\t", names=["user", "item", "rating", "timestamp"], dtype={"user": str, "item": str, "rating": float, "timestamp": int})

    unique_users = ml100k['user'].nunique()
    unique_items = ml100k['item'].nunique()
    avg_interaction_of_users = ml100k.groupby('user').size().mean()
    avg_interaction_of_items = ml100k.groupby('item').size().mean()
    avg_rating = ml100k['rating'].mean()
    sparsity = 1 - (len(ml100k) / (unique_users * unique_items))

    return unique_users, unique_items, avg_interaction_of_users, avg_interaction_of_items, avg_rating, sparsity

def ml1m_information():
    ml1m = pd.read_csv("Data/ml-1m/ratings.csv", sep=",", skiprows=1, names=["user", "item", "rating", "timestamp"], dtype={"user": str, "item": str, "rating": float, "timestamp": int})

    unique_users = ml1m['user'].nunique()
    unique_items = ml1m['item'].nunique()
    avg_interaction_of_users = ml1m.groupby('user').size().mean()
    avg_interaction_of_items = ml1m.groupby('item').size().mean()
    avg_rating = ml1m['rating'].mean()
    sparsity = 1 - (len(ml1m) / (unique_users * unique_items))

    return unique_users, unique_items, avg_interaction_of_users, avg_interaction_of_items, avg_rating, sparsity

def anime_information():
    anime = pd.read_csv("Data/anime/ratings.csv", sep=",", skiprows=1, names=["user", "item", "rating"], dtype={"user": str, "item": str, "rating": float})

    unique_users = anime['user'].nunique()
    unique_items = anime['item'].nunique()
    avg_interaction_of_users = anime.groupby('user').size().mean()
    avg_interaction_of_items = anime.groupby('item').size().mean()
    avg_rating = anime['rating'].mean()
    sparsity = 1 - (len(anime) / (unique_users * unique_items))

    return unique_users, unique_items, avg_interaction_of_users, avg_interaction_of_items, avg_rating, sparsity

def book_information():
    bookcrossing = pd.read_csv("Data/book-crossing/ratings.csv", sep=",", skiprows=1, names=["user", "item", "rating"], dtype={"user": str, "item": str, "rating": float})

    unique_users = bookcrossing['user'].nunique()
    unique_items = bookcrossing['item'].nunique()
    avg_interaction_of_users = bookcrossing.groupby('user').size().mean()
    avg_interaction_of_items = bookcrossing.groupby('item').size().mean()
    avg_rating = bookcrossing['rating'].mean()
    sparsity = 1 - (len(bookcrossing) / (unique_users * unique_items))

    return unique_users, unique_items, avg_interaction_of_users, avg_interaction_of_items, avg_rating, sparsity

ml100k = ml100k_information()
print("MovieLens 100K dataset information: ")
print("Unique users: ", ml100k[0])
print("Unique items: ", ml100k[1])
print("Average interaction of users: ", ml100k[2])
print("Average interaction of items: ", ml100k[3])
print("Average rating: ", ml100k[4])
print("Sparsity: ", ml100k[5])

print("----------------------")
      
ml1m = ml1m_information()
print("MovieLens 1M dataset information: ")
print("Unique users: ", ml1m[0])
print("Unique items: ", ml1m[1])
print("Average interaction of users: ", ml1m[2])
print("Average interaction of items: ", ml1m[3])
print("Average rating: ", ml1m[4])
print("Sparsity: ", ml1m[5])

print("----------------------")

anime = anime_information()
print("Anime dataset information: ")
print("Unique users: ", anime[0])
print("Unique items: ", anime[1])
print("Average interaction of users: ", anime[2])
print("Average interaction of items: ", anime[3])
print("Average rating: ", anime[4])
print("Sparsity: ", anime[5])

print("----------------------")

book = book_information()
print("Book-Crossing dataset information: ")
print("Unique users: ", book[0])
print("Unique items: ", book[1])
print("Average interaction of users: ", book[2])
print("Average interaction of items: ", book[3])
print("Average rating: ", book[4])
print("Sparsity: ", book[5])


