import pickle as pkl
import time
import pandas as pd
from tqdm import tqdm
from lenskit import batch
from nDCG import nDCG_LK
from lenskit.algorithms import Recommender, item_knn

"""
itemknn_lenskit_results1
########################


This script is used to calculate the nDCG for the ItemKNN algorithm on the datasets: ml100k, ml1m, anime, modcloth.
The datasets that are used have been splitted by RecBole in order to make sure, that both frameworks use exactly the same (splitted) 
data. The LensKit implemention of the similarity matrix has not been modified yet.
"""

def load_dataset(train_path, test_path):
    train = pd.read_csv(train_path, skiprows=1, sep=",", names=["user", "item", "rating"], dtype={"user": str, "item": str, "rating": float})
    test = pd.read_csv(test_path, skiprows=1, sep=",", names=["user", "item", "rating"], dtype={"user": str, "item": str, "rating": float})
    return train, test

def print_dataset_stats(train, test):
    for name, data in zip(["Train", "Test"], [train, test]):
        print(f"{name} set:")
        print("Number unique users: ", data['user'].nunique())
        print("Average actions of users: ", data.shape[0] / data['user'].nunique())
        print("The number of unique items: ", data['item'].nunique())
        print("Average actions of items: ", data.shape[0] / data['item'].nunique())
        print("The number of interactions: ", data.shape[0])
        print("The sparsity of the dataset: ", 1 - data.shape[0] / (data['user'].nunique() * data['item'].nunique()))
        print("-------------------------------------------")

def itemknn_evaluation(train_path, test_path, recommendation_path, dataset_name):
    train, test = load_dataset(train_path, test_path)
    print_dataset_stats(train, test)

    itemknn = item_knn.ItemItem(20, feedback="implicit")
    fittable = Recommender.adapt(itemknn)
    fittable.fit(train)

    users = test.user.unique()
    start_time = time.time()
    print("Start time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))

    try:
        ii_pred = pkl.load(open(recommendation_path, "rb"))
        print("Loaded recommendations from file.")
    except FileNotFoundError:
        print("Training and generating recommendations...")
        ii_pred = batch.recommend(fittable, users, 10, n_jobs=1)
        pkl.dump(ii_pred, open(recommendation_path, "wb"))

    end_time = time.time()
    print("End time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))

    if users is None or test is None:
        return 0

    total_ndcg = 0
    for user in tqdm(users, desc='Calculating nDCG', unit='user'):
        user_test_items = test[test['user'] == user]['item'].values
        user_recs = ii_pred[ii_pred['user'] == user]['item'].values
        #ndcg = calculate_ndcg(user_recs, user_test_items)
        ndcg = nDCG_LK(10, user_recs, user_test_items).calculate()
        total_ndcg += ndcg

    average_ndcg = total_ndcg / len(users)
    return average_ndcg

if __name__ == "__main__":
    datasets = {
        "ml100k_84": ("Data/Datasplits/ml-100k/trainset_ml100k_84.csv", "Data/Datasplits/ml-100k/testset_ml100k_84.csv", "Recommendations_SameData_SameNDCG/ml100k_84_implicit.pkl"),
        "ml100k_42": ("Data/Datasplits/ml-100k/trainset_ml100k_42.csv", "Data/Datasplits/ml-100k/testset_ml100k_42.csv", "Recommendations_SameData_SameNDCG/ml100k_42_implicit.pkl"),
        "ml100k_21": ("Data/Datasplits/ml-100k/trainset_ml100k_21.csv", "Data/Datasplits/ml-100k/testset_ml100k_21.csv", "Recommendations_SameData_SameNDCG/ml100k_21_implicit.pkl"),
        "ml1m_84": ("Data/Datasplits/ml-1m/trainset_ml1m_84.csv", "Data/Datasplits/ml-1m/testset_ml1m_84.csv", "Recommendations_SameData_SameNDCG/ml1m_84_implicit.pkl"),
        "ml1m_42": ("Data/Datasplits/ml-1m/trainset_ml1m_42.csv", "Data/Datasplits/ml-1m/testset_ml1m_42.csv", "Recommendations_SameData_SameNDCG/ml1m_42_implicit.pkl"),
        "ml1m_21" : ("Data/Datasplits/ml-1m/trainset_ml1m_21.csv", "Data/Datasplits/ml-1m/testset_ml1m_21.csv", "Recommendations_SameData_SameNDCG/ml1m_21_implicit.pkl"),
        "modcloth_84": ("Data/Datasplits/modcloth/trainset_modcloth_84.csv", "Data/Datasplits/modcloth/testset_modcloth_84.csv", "Recommendations_SameData_SameNDCG/modcloth_84_implicit.pkl"),
        "modcloth_42": ("Data/Datasplits/modcloth/trainset_modcloth_42.csv", "Data/Datasplits/modcloth/testset_modcloth_42.csv", "Recommendations_SameData_SameNDCG/modcloth_42_implicit.pkl"),
        "modcloth_21": ("Data/Datasplits/modcloth/trainset_modcloth_21.csv", "Data/Datasplits/modcloth/testset_modcloth_21.csv", "Recommendations_SameData_SameNDCG/modcloth_21_implicit.pkl"),
        "anime_84": ("Data/Datasplits/anime/trainset_anime_84.csv", "Data/Datasplits/anime/testset_anime_84.csv", "Recommendations_SameData_SameNDCG/anime_84_implicit.pkl"),
        "anime_42": ("Data/Datasplits/anime/trainset_anime_42.csv", "Data/Datasplits/anime/testset_anime_42.csv", "Recommendations_SameData_SameNDCG/anime_42_implicit.pkl"),
        "anime_21": ("Data/Datasplits/anime/trainset_anime_21.csv", "Data/Datasplits/anime/testset_anime_21.csv", "Recommendations_SameData_SameNDCG/anime_21_implicit.pkl"),
    }

    for name, paths in datasets.items():
        train_path, test_path, recommendation_path = paths
        ndcg = itemknn_evaluation(train_path, test_path, recommendation_path, name)
        print(f"NDCG for {name}: {ndcg}")