import pickle as pkl
import time
import pandas as pd
from tqdm import tqdm
from lenskit import batch
from nDCG import nDCG_LK
from lenskit.algorithms import Recommender, item_knn


"""
itemknn_lenskit_results2
########################


This script is used to calculate the nDCG for the ItemKNN algorithm on the datasets: ml100k, ml1m, anime, modcloth.
The datasets that are used have been splitted by RecBole in order to make sure, that both frameworks use exactly the same (splitted) 
data. 
The LensKit implemention of the similarity matrix has now been modified. For each item, the similarity matrix now stores only the topN items. 
A deep explanation on the differences between the two implementations can be found in our paper.
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

def ndcg_evaluation(train_path, test_path, recommendation_path, dataset_name):
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
        ndcg = nDCG_LK(10, user_recs, user_test_items).calculate()
        total_ndcg += ndcg

    average_ndcg = total_ndcg / len(users)
    return average_ndcg

if __name__ == "__main__":
    datasets = {
        "ml100k": ("Data/Datasplits/ml-100k/trainset_ml100k.csv", "Data/Datasplits/ml-100k/testset_ml100k.csv", "Recommendations_customized_ItemKNN/saved_recommendations_ml100k_implicit.pkl"),
        "ml1m": ("Data/Datasplits/ml-1m/trainset_ml-1m.csv", "Data/Datasplits/ml-1m/testset_ml-1m.csv", "Recommendations_customized_ItemKNN/saved_recommendations_ml1m_implicit.pkl"),
        "anime": ("Data/Datasplits/anime/trainset_anime.csv", "Data/Datasplits/anime/testset_anime.csv", "Recommendations_customized_ItemKNN/saved_recommendations_anime_implicit.pkl"),
        # Bookcrossing dataset cant be used
        # "bookcrossing": ("Data/Datasplits/book-crossing/trainset_bookcrossing.csv", "Data/Datasplits/book-crossing/testset_bookcrossing.csv", "saved_recommendations_bookcrossing_implicit.pkl"),
        "modcloth": ("Data/Datasplits/modcloth/trainset_modcloth.csv", "Data/Datasplits/modcloth/testset_modcloth.csv", "Recommendations_customized_ItemKNN/saved_recommendations_modcloth_implicit.pkl")
    }

    for name, paths in datasets.items():
        train_path, test_path, recommendation_path = paths
        ndcg = ndcg_evaluation(train_path, test_path, recommendation_path, name)
        print(f"NDCG for {name}: {ndcg}")