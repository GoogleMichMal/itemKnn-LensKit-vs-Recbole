import pickle as pkl
import os
import pandas as pd
from lenskit import batch
from lenskit.algorithms import Recommender, item_knn
from tqdm import tqdm
from calculate_ndcg import calculate_ndcg
from lk_partition_user import lk_partition_users

def lenskit_ml1m_ownNDCG():
    recs_file = "saved_recommendations_ml1m.pkl"
    users = None
    test = None

    try:
        # Read the recommendations from file
        recs = pkl.load(open(recs_file, "rb"))
        print("Loaded recommendations from file.")
        # Get the Users who have recommendations
        users = recs['user'].unique()
        print("Users who have recommendations: ", len(users))

        # Read the Truth dataset
        ml1m = pd.read_csv("Data/ml-1m/ratings.csv", sep=",", skiprows=1, names=["user", "item", "rating", "timestamp"], dtype={"user": str, "item": str, "rating": float, "timestamp": int})
        num_unique_users = ml1m['user'].nunique()
        num_unique_items = ml1m['item'].nunique()
        print("Number of unique users:", num_unique_users)
        print("Number of unique items:", num_unique_items)
        train, test = lk_partition_users(ml1m)

        # test = ml1m.sample(frac=0.2, random_state=42)

    except FileNotFoundError:
        print(f"File '{recs_file}' not found. Training and generating recommendations...")
        ml1m = pd.read_csv("Data/ml-1m/ratings.csv", sep=",", skiprows=1, names=["user", "item", "rating", "timestamp"], dtype={"user": str, "item": str, "rating": float, "timestamp": int})
        print("Number of unique users:", ml1m['user'].nunique())
        train, test = lk_partition_users(ml1m)


        itemknn = item_knn.ItemItem(20, feedback="implicit")
        fittable = Recommender.adapt(itemknn)
        fittable.fit(train)

        users = test['user'].unique()
        print("Number of unique users:", len(users))
        recs = batch.recommend(fittable, users, 10, n_jobs=1)
        pkl.dump(recs, open(recs_file, "wb"))  # Save recommendations to file
        print("Recommendations saved to file.")

    total_ndcg = 0
    for user in tqdm(users, desc='Calculating nDCG', unit='user'):
        # Return the items in the test set for the user
        user_test_items = test[test['user'] == user]['item'].values
        # print("User test items: ", user_test_items)
        # Return the items recommended to the user
        user_recs = recs[recs['user'] == user]['item'].values
        # print("User recommendations: ", user_recs)
        ndcg = calculate_ndcg(user_recs, user_test_items)
        # print("nDCG for user", user, ":", ndcg)
        total_ndcg += ndcg

    average_ndcg = total_ndcg / len(users)
    return average_ndcg

print(lenskit_ml1m_ownNDCG())
