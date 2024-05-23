import pickle as pkl
import time
import pandas as pd
from tqdm import tqdm
from lk_partition_users import lk_partition_users
from make_implicit_movielens import make_implicit_movielens
from make_implicit_anime_bookcrossing import make_implicit_anime_bookcrossing
from lenskit import batch
from calculate_ndcg import calculate_ndcg
from lenskit.algorithms import Recommender, item_knn


def ndcg_ml100k():
    # ml100k = pd.read_csv("Data/ml-100k/u.data", header=None, sep="\t", names=["user", "item", "rating", "timestamp"])

    # data = make_implicit_movielens(ml100k)
    # print("Number of interactions in the dataset: ", data.shape[0])

    # train, test = lk_partition_users(data)

    train = pd.read_csv("Data/Datasplits/ml-100k/trainset_ml100k.csv", skiprows=1, sep=",", names=["user", "item", "rating"], dtype={"user": str, "item": str, "rating": float})
    test = pd.read_csv("Data/Datasplits/ml-100k/testset_ml100k.csv", skiprows=1, sep=",", names=["user", "item", "rating"], dtype={"user": str, "item": str, "rating": float})

    print("Train set:")
    print("Number unique users: ", train['user'].nunique())
    print("Average actions of users: ", train.shape[0] / train['user'].nunique())
    print("The number of unique items: ", train['item'].nunique())
    print("Average actions of items: ", train.shape[0] / train['item'].nunique())
    print("The number of inters: ", train.shape[0])
    print("The sparsity of the dataset: ", 1 - train.shape[0] / (train['user'].nunique() * train['item'].nunique()))
    print("-------------------------------------------")
    print("Test set:")
    print("Number unique users: ", test['user'].nunique())
    print("Average actions of users: ", test.shape[0] / test['user'].nunique())
    print("The number of unique items: ", test['item'].nunique())
    print("Average actions of items: ", test.shape[0] / test['item'].nunique())
    print("The number of inters: ", test.shape[0])
    print("The sparsity of the dataset: ", 1 - test.shape[0] / (test['user'].nunique() * test['item'].nunique()))

    itemknn = item_knn.ItemItem(20,feedback="implicit")

    fittable = Recommender.adapt(itemknn)

    fittable.fit(train)

    users = test.user.unique()

    start_time = time.time()
        
    formatted_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    print("Start time:", formatted_start_time)


    try:
        ii_pred = pkl.load(open("saved_recommendations_ml100k_implicit.pkl", "rb"))
        print("Loaded recommendations from file.")

    except FileNotFoundError:
        print("Training and generating recommendations...")
        ii_pred = batch.recommend(fittable, users, 10, n_jobs=1)
        pkl.dump(ii_pred, open(f"saved_recommendations_ml100k_implicit.pkl", "wb"))

    end_time = time.time()
    formatted_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    print("Start time:", formatted_start_time)

    if users is None or test is None:
        return 0

    total_ndcg = 0
    for user in tqdm(users, desc='Calculating nDCG', unit='user'):
        # Return the items in the test set for the user
        user_test_items = test[test['user'] == user]['item'].values
        # print("User test items", user, ":", user_test_items)
        # Return the items recommended to the user
        user_recs = ii_pred[ii_pred['user'] == user]['item'].values
        # print("User recommendations", user, ":", user_recs)
        ndcg = calculate_ndcg(user_recs, user_test_items)
        # print("nDCG for user", user, ":", ndcg)
        total_ndcg += ndcg

    average_ndcg = total_ndcg / len(users)
    return average_ndcg

# print("NDCG for ml-100k: ", ndcg_ml100k())

def ndcg_ml1m():
    # ml1m = pd.read_csv("Data/ml-1m/ratings.csv", sep=",", skiprows=1, names=["user", "item", "rating", "timestamp"], dtype={"user": str, "item": str, "rating": float, "timestamp": int})

    # data = make_implicit_movielens(ml1m)
    # print("Number of interactions in the dataset: ", data.shape[0])

    # train, test = lk_partition_users(data)

    train = pd.read_csv("Data/Datasplits/ml-1m/trainset_ml-1m.csv", skiprows=1, sep=",", names=["user", "item", "rating"], dtype={"user": str, "item": str, "rating": float})
    test = pd.read_csv("Data/Datasplits/ml-1m/testset_ml-1m.csv", skiprows=1, sep=",", names=["user", "item", "rating"], dtype={"user": str, "item": str, "rating": float})

    print("Train set:")
    print("Number unique users: ", train['user'].nunique())
    print("Average actions of users: ", train.shape[0] / train['user'].nunique())
    print("The number of unique items: ", train['item'].nunique())
    print("Average actions of items: ", train.shape[0] / train['item'].nunique())
    print("The number of inters: ", train.shape[0])
    print("The sparsity of the dataset: ", 1 - train.shape[0] / (train['user'].nunique() * train['item'].nunique()))
    print("-------------------------------------------")
    print("Test set:")
    print("Number unique users: ", test['user'].nunique())
    print("Average actions of users: ", test.shape[0] / test['user'].nunique())
    print("The number of unique items: ", test['item'].nunique())
    print("Average actions of items: ", test.shape[0] / test['item'].nunique())
    print("The number of inters: ", test.shape[0])
    print("The sparsity of the dataset: ", 1 - test.shape[0] / (test['user'].nunique() * test['item'].nunique()))

    itemknn = item_knn.ItemItem(20,feedback="implicit")

    fittable = Recommender.adapt(itemknn)

    fittable.fit(train)

    users = test.user.unique()

    start_time = time.time()
        
    formatted_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    print("Start time:", formatted_start_time)


    try:
        ii_pred = pkl.load(open("saved_recommendations_ml1m_implicit.pkl", "rb"))
        print("Loaded recommendations from file.")

    except FileNotFoundError:
        print("Training and generating recommendations...")
        ii_pred = batch.recommend(fittable, users, 10, n_jobs=1)
        pkl.dump(ii_pred, open(f"saved_recommendations_ml1m_implicit.pkl", "wb"))

    end_time = time.time()
    formatted_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    print("Start time:", formatted_start_time)

    if users is None or test is None:
        return 0

    total_ndcg = 0
    for user in tqdm(users, desc='Calculating nDCG', unit='user'):
        # Return the items in the test set for the user
        user_test_items = test[test['user'] == user]['item'].values
        # print("User test items", user, ":", user_test_items)
        # Return the items recommended to the user
        user_recs = ii_pred[ii_pred['user'] == user]['item'].values
        # print("User recommendations", user, ":", user_recs)
        ndcg = calculate_ndcg(user_recs, user_test_items)
        # print("nDCG for user", user, ":", ndcg)
        total_ndcg += ndcg

    average_ndcg = total_ndcg / len(users)
    return average_ndcg

# print("NDCG for ml-1m: ", ndcg_ml1m())

def ndcg_anime():
    # anime = pd.read_csv("Data/anime/ratings.csv", sep=",", skiprows=1, names=["user", "item", "rating"], dtype={"user": str, "item": str, "rating": float})

    # data = make_implicit_anime_bookcrossing(anime)
    # print("Number of interactions in the dataset: ", data.shape[0])

    # train, test = lk_partition_users(data)

    train = pd.read_csv("Data/Datasplits/anime/trainset_anime.csv", skiprows=1, sep=",", names=["user", "item", "rating"], dtype={"user": str, "item": str, "rating": float})
    test = pd.read_csv("Data/Datasplits/anime/testset_anime.csv", skiprows=1, sep=",", names=["user", "item", "rating"], dtype={"user": str, "item": str, "rating": float})

    print("Train set:")
    print("Number unique users: ", train['user'].nunique())
    print("Average actions of users: ", train.shape[0] / train['user'].nunique())
    print("The number of unique items: ", train['item'].nunique())
    print("Average actions of items: ", train.shape[0] / train['item'].nunique())
    print("The number of inters: ", train.shape[0])
    print("The sparsity of the dataset: ", 1 - train.shape[0] / (train['user'].nunique() * train['item'].nunique()))
    print("-------------------------------------------")
    print("Test set:")
    print("Number unique users: ", test['user'].nunique())
    print("Average actions of users: ", test.shape[0] / test['user'].nunique())
    print("The number of unique items: ", test['item'].nunique())
    print("Average actions of items: ", test.shape[0] / test['item'].nunique())
    print("The number of inters: ", test.shape[0])
    print("The sparsity of the dataset: ", 1 - test.shape[0] / (test['user'].nunique() * test['item'].nunique()))

    itemknn = item_knn.ItemItem(20,feedback="implicit")

    fittable = Recommender.adapt(itemknn)

    fittable.fit(train)

    users = test.user.unique()

    start_time = time.time()
        
    formatted_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    print("Start time:", formatted_start_time)


    try:
        ii_pred = pkl.load(open("saved_recommendations_anime_implicit.pkl", "rb"))
        print("Loaded recommendations from file.")

    except FileNotFoundError:
        print("Training and generating recommendations...")
        ii_pred = batch.recommend(fittable, users, 10, n_jobs=1)
        pkl.dump(ii_pred, open(f"saved_recommendations_anime_implicit.pkl", "wb"))

    end_time = time.time()
    formatted_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    print("Start time:", formatted_start_time)

    if users is None or test is None:
        return 0

    total_ndcg = 0
    for user in tqdm(users, desc='Calculating nDCG', unit='user'):
        # Return the items in the test set for the user
        user_test_items = test[test['user'] == user]['item'].values
        # print("User test items", user, ":", user_test_items)
        # Return the items recommended to the user
        user_recs = ii_pred[ii_pred['user'] == user]['item'].values
        # print("User recommendations", user, ":", user_recs)
        ndcg = calculate_ndcg(user_recs, user_test_items)
        # print("nDCG for user", user, ":", ndcg)
        total_ndcg += ndcg

    average_ndcg = total_ndcg / len(users)
    return average_ndcg

# print("NDCG for anime:", ndcg_anime())

def ndcg_bookcrossing():
    # bookcrossing = pd.read_csv("Data/book-crossing/ratings.csv", sep=",", skiprows=1, names=["user", "item", "rating"], dtype={"user": str, "item": str, "rating": float})

    # data = make_implicit_anime_bookcrossing(bookcrossing)
    # print("Number of interactions in the dataset: ", data.shape[0])

    # train, test = lk_partition_users(data)

    train = pd.read_csv("Data/Datasplits/book-crossing/trainset_bookcrossing.csv", skiprows=1, sep=",", names=["user", "item", "rating"], dtype={"user": str, "item": str, "rating": float})
    test = pd.read_csv("Data/Datasplits/book-crossing/testset_bookcrossing.csv", skiprows=1, sep=",", names=["user", "item", "rating"], dtype={"user": str, "item": str, "rating": float})

    print("Train set:")
    print("Number unique users: ", train['user'].nunique())
    print("Average actions of users: ", train.shape[0] / train['user'].nunique())
    print("The number of unique items: ", train['item'].nunique())
    print("Average actions of items: ", train.shape[0] / train['item'].nunique())
    print("The number of inters: ", train.shape[0])
    print("The sparsity of the dataset: ", 1 - train.shape[0] / (train['user'].nunique() * train['item'].nunique()))
    print("-------------------------------------------")
    print("Test set:")
    print("Number unique users: ", test['user'].nunique())
    print("Average actions of users: ", test.shape[0] / test['user'].nunique())
    print("The number of unique items: ", test['item'].nunique())
    print("Average actions of items: ", test.shape[0] / test['item'].nunique())
    print("The number of inters: ", test.shape[0])
    print("The sparsity of the dataset: ", 1 - test.shape[0] / (test['user'].nunique() * test['item'].nunique()))

    itemknn = item_knn.ItemItem(20,feedback="implicit")

    fittable = Recommender.adapt(itemknn)

    fittable.fit(train)

    users = test.user.unique()

    start_time = time.time()
        
    formatted_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    print("Start time:", formatted_start_time)


    try:
        ii_pred = pkl.load(open("saved_recommendations_bookcrossing_implicit.pkl", "rb"))
        print("Loaded recommendations from file.")

    except FileNotFoundError:
        print("Training and generating recommendations...")
        ii_pred = batch.recommend(fittable, users, 10, n_jobs=1)
        pkl.dump(ii_pred, open(f"saved_recommendations_bookcrossing_implicit.pkl", "wb"))

    end_time = time.time()
    formatted_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    print("Start time:", formatted_start_time)

    if users is None or test is None:
        return 0

    total_ndcg = 0
    for user in tqdm(users, desc='Calculating nDCG', unit='user'):
        # Return the items in the test set for the user
        user_test_items = test[test['user'] == user]['item'].values
        # print("User test items", user, ":", user_test_items)
        # Return the items recommended to the user
        user_recs = ii_pred[ii_pred['user'] == user]['item'].values
        # print("User recommendations", user, ":", user_recs)
        ndcg = calculate_ndcg(user_recs, user_test_items)
        # print("nDCG for user", user, ":", ndcg)
        total_ndcg += ndcg

    average_ndcg = total_ndcg / len(users)
    return average_ndcg

# print("NDCG for bookcrossing:", ndcg_bookcrossing())