import numpy as np
import pickle as pkl
import pandas as pd
from lenskit.datasets import ML100K
from lenskit import batch, topn
from lenskit.algorithms import Recommender, item_knn
from tqdm import tqdm
from calculate_ndcg import calculate_ndcg

def bookcrossing_ownndcg():
    users = None
    test = None
    try:
        recs_df = pkl.load(open("saved_recommendations_bookcrossing.pkl", "rb"))
        print("Loaded recommendations from file.")
        users = recs_df['user'].unique()

        bookcrossing = pd.read_csv("Data/book-crossing/ratings.csv", sep=";", skiprows=1, names=["user", "item", "rating"], dtype={"user": str, "item": str, "rating": float})
        num_unique_users = bookcrossing['user'].nunique()
        num_unique_items = bookcrossing['item'].nunique()
        print("Number of unique users:", num_unique_users)
        print("Number of unique items:", num_unique_items)
        test = bookcrossing.sample(frac=0.2, random_state=42)

    except FileNotFoundError:
        print("Training and generating recommendations...")
        # Read the Book Crossing dataset
        bookcrossing = pd.read_csv("Data/book-crossing/ratings.csv", sep=";", skiprows=1, names=["user", "item", "rating"], dtype={"user": str, "item": str, "rating": float})
        num_unique_users = bookcrossing['user'].nunique()
        num_unique_items = bookcrossing['item'].nunique()
        print("Number of unique users:", num_unique_users)
        print("Number of unique items:", num_unique_items)

        # Split the data into training and test sets
        test = bookcrossing.sample(frac=0.2, random_state=42)
        train = bookcrossing.drop(test.index)

        # Create an ItemItem recommender
        itemknn = item_knn.ItemItem(20, feedback="implicit")

        fittable = Recommender.adapt(itemknn)
        fittable.fit(train)

        # Get the unique users
        users = test.user.unique()

        recs = []
        for user in tqdm(users, desc="Generating Recommendations"):
            recs.append(batch.recommend(fittable, [user], 10, n_jobs=1).assign(user=user))

        # Combine the recommendations into a single DataFrame
        recs_df = pd.concat(recs, ignore_index=True)  

        # Save recommendations to file
        pkl.dump(recs_df, open("saved_recommendations_bookcrossing.pkl", "wb"))
        print("Recommendations saved to file.")

    if users is None or test is None:
        return 0

    # Calculate nDCG
    total_ndcg = 0
    for user in tqdm(users, desc='Calculating nDCG', unit='user'):
        user_test_items = test[test['user'] == user]['item'].values
        user_recs = recs_df[recs_df['user'] == user]['item'].values
        ndcg = calculate_ndcg(user_recs, user_test_items)
        total_ndcg += ndcg

    # Calculate average nDCG over all users
    average_ndcg = total_ndcg / len(users)
    return average_ndcg

def itemknn_lenskit_bookcrossing_ownndcg():
    # Read the Book-Crossing dataset
    bookcrossing = pd.read_csv("Data/book-crossing/ratings.csv", sep=";", skiprows=1, names=["user", "item", "rating"], dtype={"user": str, "item": str, "rating": float})

    # Print out the number of unique users and items
    num_unique_users = bookcrossing['user'].nunique()
    num_unique_items = bookcrossing['item'].nunique()
    print("Number of unique users:", num_unique_users)
    print("Number of unique items:", num_unique_items)


    # Split the data into training and test sets
    test = bookcrossing.sample(frac=0.2, random_state=42)
    train = bookcrossing.drop(test.index)

    # Create an ItemItem recommender
    itemknn = item_knn.ItemItem(20, feedback="implicit")

    # Adapt the recommender to the LensKit API
    fittable = Recommender.adapt(itemknn)
    fittable.fit(train)

    # Get the unique users
    users = list(test.user.unique())  

    recs = []
    for user in tqdm(users, desc="Generating Recommendations"):
        recs.append(batch.recommend(fittable, [user], 10, n_jobs=1).assign(user=user))

    # Combine the recommendations into a single DataFrame
    recs_df = pd.concat(recs, ignore_index=True)  

    # Calculate nDCG
    total_ndcg = 0
    with tqdm(total=len(users), desc='Calculating nDCG', unit='user') as pbar:
        for user in users:
            user_test_items = test[test['user'] == user]['item'].values
            user_recs = recs_df[recs_df['user'] == user]['item'].values
            ndcg = calculate_ndcg(user_recs, user_test_items)
            total_ndcg += ndcg
            pbar.update(1)

    # Calculate average nDCG over all users
    average_ndcg = total_ndcg / len(users)
    return average_ndcg

def itemknn_lenskit_food():
    food = pd.read_csv("Data/food/RAW_interactions.csv", sep=",", quotechar='"', skiprows=1, usecols=[0,1,3], names=["user", "item", "rating"], dtype={"user": str, "item": str, "rating": float})
    
    num_unique_users = food['user'].nunique()
    num_unique_items = food['item'].nunique()
    print("Number of unique users:", num_unique_users)
    print("Number of unique items:", num_unique_items)

    test = food.sample(frac=0.2, random_state=42)
    train = food.drop(test.index)

    itemknn = item_knn.ItemItem(20, feedback="implicit")

    fittable = Recommender.adapt(itemknn)
    fittable.fit(train)

    # List of unique users
    users = list(test.user.unique())  

    recs = []
    for user in tqdm(users, desc="Generating Recommendations"):
        recs.append(batch.recommend(fittable, [user], 10, n_jobs=1).assign(user=user))

    # Combine the recommendations into a single DataFrame
    recs_df = pd.concat(recs, ignore_index=True)  

    # Initialize the RecListAnalysis object
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    rla.add_metric(topn.precision)
    rla.add_metric(topn.recall)

    # Compute the evaluation metrics
    results = rla.compute(recs_df, test)
    return results.mean()


def itemknn_lenskit_ml20m_ownndcg():
    # Read the Book-Crossing dataset
    ml20m = pd.read_csv("Data/ml-20m/ratings.csv", sep=",", skiprows=1, names=["user", "item", "rating", "timestamp"], dtype={"user": str, "item": str, "rating": float, "timestamp": int})
    ml20m = ml20m.drop(columns=["timestamp"])

    # Print out the number of unique users and items
    num_unique_users = ml20m['user'].nunique()
    num_unique_items = ml20m['item'].nunique()
    print("Number of unique users:", num_unique_users)
    print("Number of unique items:", num_unique_items)


    # Split the data into training and test sets
    test = ml20m.sample(frac=0.2, random_state=42)
    train = ml20m.drop(test.index)

    # Create an ItemItem recommender
    itemknn = item_knn.ItemItem(20, feedback="implicit")

    # Adapt the recommender to the LensKit API
    fittable = Recommender.adapt(itemknn)
    fittable.fit(train)

    # Get the unique users
    users = list(test.user.unique())  

    recs = []
    for user in tqdm(users, desc="Generating Recommendations"):
        recs.append(batch.recommend(fittable, [user], 10, n_jobs=1).assign(user=user))

    # Combine the recommendations into a single DataFrame
    recs_df = pd.concat(recs, ignore_index=True)  

    # Calculate nDCG
    total_ndcg = 0
    with tqdm(total=len(users), desc='Calculating nDCG', unit='user') as pbar:
        for user in users:
            user_test_items = test[test['user'] == user]['item'].values
            user_recs = recs_df[recs_df['user'] == user]['item'].values
            ndcg = calculate_ndcg(user_recs, user_test_items)
            total_ndcg += ndcg
            pbar.update(1)

    # Calculate average nDCG over all users
    average_ndcg = total_ndcg / len(users)
    return average_ndcg


# Old implementations
'''
def itemknn_lenskit_ml100k():
    # Read the MovieLens 100K dataset
    ml100k = ML100K("data/ml-100k")
    ratings = ml100k.ratings
    ratings.head()

    # Split the data into training and test sets
    test = ratings.sample(frac=0.2, random_state=42)
    train = ratings.drop(test.index)

    itemknn = item_knn.ItemItem(20, feedback="implicit")

    fittable = Recommender.adapt(itemknn)

    fittable.fit(train)

    users = test.user.unique()
    # print("User 815: ", test.loc[test['user'] == users[1]])

    # Gibt 10 Recommendations für die User aus
    recs = batch.recommend(fittable, users, 10, n_jobs=1)
    # print("Recommendations for User an der Stelle [1]", recs.loc[recs['user'] == users[1]])

    total_ndcg = 0
    for user in users:
        # Return the items in the test set for the user
        user_test_items = test[test['user'] == user]['item'].values
        print("User test items: ", user_test_items)
        # Return the items recommended to the user
        user_recs = recs[recs['user'] == user]['item'].values
        print("User recommendations: ", user_recs)
        ndcg = calculate_ndcg(user_recs, user_test_items)
        print("nDCG for user", user, ":", ndcg)
        total_ndcg += ndcg

    average_ndcg = total_ndcg / len(users)
    return average_ndcg

print(itemknn_lenskit_ml100k())
'''


'''
def itemknn_lenskit_bookcrossing():
    # Read the Book-Crossing dataset
    bookcrossing = pd.read_csv("Data/book-crossing/Ratings.csv", sep=";", skiprows=1, names=["user", "item", "rating"], dtype={"user": str, "item": str, "rating": float})

    # Print out the number of unique users and items
    num_unique_users = bookcrossing['user'].nunique()
    num_unique_items = bookcrossing['item'].nunique()
    print("Number of unique users:", num_unique_users)
    print("Number of unique items:", num_unique_items)


    # Split the data into training and test sets
    test = bookcrossing.sample(frac=0.2, random_state=42)
    train = bookcrossing.drop(test.index)

    # Create an ItemItem recommender
    itemknn = item_knn.ItemItem(20, feedback="implicit")

    # Adapt the recommender to the LensKit API
    fittable = Recommender.adapt(itemknn)
    fittable.fit(train)

    # Get the unique users
    users = list(test.user.unique())  

    recs = []
    for user in tqdm(users, desc="Generating Recommendations"):
        recs.append(batch.recommend(fittable, [user], 10, n_jobs=1).assign(user=user))

    # Combine the recommendations into a single DataFrame
    recs_df = pd.concat(recs, ignore_index=True)  

    # Initialize the RecListAnalysis object
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    rla.add_metric(topn.precision)
    rla.add_metric(topn.recall)

    # Compute the evaluation metrics
    results = rla.compute(recs_df, test)
    return results.mean()
'''