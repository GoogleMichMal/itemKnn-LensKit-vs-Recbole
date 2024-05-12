import pickle as pkl
import pandas as pd
from lenskit import batch
from lenskit.algorithms import Recommender, item_knn
from tqdm import tqdm
from calculate_ndcg import calculate_ndcg


def lenskit_bookcrossing_ownNDCG():
    users = None
    test = None
    recs = None
    try:
        # Read the recommendations from file
        recs_df = pkl.load(open("saved_recommendations_bookcrossing.pkl", "rb"))
        print("Loaded recommendations from file.")
        # Get the Users who have recommendations
        users = recs_df['user'].unique()

        # Read the Truth dataset
        bookcrossing = pd.read_csv("Data/book-crossing/ratings.csv", sep=";", skiprows=1, names=["user", "item", "rating"], dtype={"user": str, "item": str, "rating": float})
        num_unique_users = bookcrossing['user'].nunique()
        num_unique_items = bookcrossing['item'].nunique()
        print("Number of unique users:", num_unique_users)
        print("Number of unique items:", num_unique_items)
        test = bookcrossing.sample(frac=0.2, random_state=42)

        recs = [recs_df]

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
        users = list(test.user.unique())  

        recs = []
        for user in tqdm(users, desc="Generating Recommendations"):
            recs.append(batch.recommend(fittable, [user], 10, n_jobs=1).assign(user=user))

        # Combine the recommendations into a single DataFrame
        recs_df = pd.concat(recs, ignore_index=True)  

        # Save recommendations to file
        pkl.dump(recs_df, open("saved_recommendations_bookcrossing.pkl", "wb"))
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

print(lenskit_bookcrossing_ownNDCG())