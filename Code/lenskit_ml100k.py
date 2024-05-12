import pickle as pkl
from lenskit.datasets import ML100K
from lenskit import batch
from lenskit.algorithms import Recommender, item_knn
from tqdm import tqdm
from calculate_ndcg import calculate_ndcg


def lenskit_ml100k_ownNDCG():
    users = None
    test = None
    try:
        recs = pkl.load(open("saved_recommendations_ml100k.pkl", "rb"))
        print("Loaded recommendations from file.")
        users = recs['user'].unique()

        ml100k = ML100K("data/ml-100k")
        ratings = ml100k.ratings
        test = ratings.sample(frac=0.2, random_state=42)

    except FileNotFoundError:
        print("Training and generating recommendations...")
        # Read the MovieLens 100K dataset
        ml100k = ML100K("data/ml-100k")
        ratings = ml100k.ratings

        # Split the data into training and test sets
        test = ratings.sample(frac=0.2, random_state=42)
        train = ratings.drop(test.index)

        itemknn = item_knn.ItemItem(20, feedback="implicit")

        fittable = Recommender.adapt(itemknn)

        fittable.fit(train)

        users = test.user.unique()

        # Generate recommendations for all users
        recs = batch.recommend(fittable, users, 10, n_jobs=1)

        # Save recommendations to file
        pkl.dump(recs, open("saved_recommendations_ml100k.pkl", "wb"))
        print("Recommendations saved to file.")

    if users is None or test is None:
        return 0

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

print(lenskit_ml100k_ownNDCG())