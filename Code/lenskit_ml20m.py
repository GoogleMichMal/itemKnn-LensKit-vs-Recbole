import pickle as pkl
import pandas as pd
from lenskit import batch
from lenskit.algorithms import Recommender, item_knn
from tqdm import tqdm
from calculate_ndcg import calculate_ndcg

def lenskit_ml20m_ownNDCG():
    users = None
    test = None
    recs = None
    try:
        # Read the recommendations from file
        recs_df = pkl.load(open("saved_recommendations_ml20m.pkl", "rb"))
        print("Loaded recommendations from file.")
        # Get the Users who have recommendations
        users = recs_df['user'].unique()

        # Read the Truth dataset
        ml20m = pd.read_csv("Data/ml-20m/ratings.csv", sep=",", skiprows=1, names=["user", "item", "rating", "timestamp"], dtype={"user": str, "item": str, "rating": float, "timestamp": int})
        ml20m = ml20m.drop(columns=[ 'timestamp'])
        num_unique_users = ml20m['user'].nunique()
        num_unique_items = ml20m['item'].nunique()
        print("Number of unique users:", num_unique_users)
        print("Number of unique items:", num_unique_items)
        test = ml20m.sample(frac=0.2, random_state=42)

        recs = [recs_df]

    except FileNotFoundError:
        print("Training and generating recommendations...")
        # Read the Book Crossing dataset
        ml20m = pd.read_csv("Data/ml-20m/ratings.csv", sep=",", skiprows=1, names=["user", "item", "rating", "timestamp"], dtype={"user": str, "item": str, "rating": float, "timestamp": int})
        ml20m = ml20m.drop(columns=[ 'timestamp'])
        num_unique_users = ml20m['user'].nunique()
        num_unique_items = ml20m['item'].nunique()
        print("Number of unique users:", num_unique_users)
        print("Number of unique items:", num_unique_items)

        # Split the data into training and test sets
        test = ml20m.sample(frac=0.2, random_state=42)
        train = ml20m.drop(test.index)

        # Create an ItemItem recommender
        itemknn = item_knn.ItemItem(20, feedback="implicit")

        fittable = Recommender.adapt(itemknn)
        fittable.fit(train)

        # Get the unique users
        users = list(test.user.unique())  

        recs = []
        for user in tqdm(users, desc="Generating Recommendations"):
            recs.append(batch.recommend(fittable, [user], 10, n_jobs=1).assign(user=user))

        # Combine the recommendations into a single DataFrame
        recs_df = pd.concat(recs, ignore_index=True)  

        # Save recommendations to file
        pkl.dump(recs_df, open("saved_recommendations_ml20m.pkl", "wb"))
        print("Recommendations saved to file.")

    if users is None or test is None or recs is None:
        return 0

    # Calculate nDCG
    total_ndcg = 0
    for user in tqdm(users, desc='Calculating nDCG', unit='user'):
        # Return the items in the test set for the user
        user_test_items = test[test['user'] == user]['item'].values
        # print("User test items: ", user_test_items)
        # Return the items recommended to the user
        user_recs = pd.concat(recs)[pd.concat(recs)['user'] == user]['item'].values
        # print("User recommendations: ", user_recs)
        ndcg = calculate_ndcg(user_recs, user_test_items)
        # print("nDCG for user", user, ":", ndcg)
        total_ndcg += ndcg

    # Calculate average nDCG over all users
    average_ndcg = total_ndcg / len(users)
    return average_ndcg

print(lenskit_ml20m_ownNDCG())