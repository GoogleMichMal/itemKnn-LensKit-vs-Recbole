import pandas as pd
from lenskit import batch, topn
from lenskit.algorithms import Recommender, item_knn
from lenskit.metrics.topn import ndcg, precision, recall
from Code.lk_partition_users import lk_partition_users
from Code.make_implicit_movielens import make_implicit_movielens

def itemknn_lenskit_ml100k():
    # Read the MovieLens 100K dataset
    ml100k = pd.read_csv("Data/ml-100k/u.data", header=None, sep="\t", names=["user", "item", "rating", "timestamp"])

    # data = make_implicit_movielens(ml100k)

    # Split the data into training and test sets
    train, test = lk_partition_users(ml100k)
    print("Train set: ", train)
    print("Test set: ", test)

    itemknn = item_knn.ItemItem(20, feedback="implicit")

    fittable = Recommender.adapt(itemknn)

    fittable.fit(train)

    users = test.user.unique()

    recs = batch.recommend(fittable, users, 10, n_jobs=1)

    # Calculate nDCG
    rla = topn.RecListAnalysis()
    rla.add_metric(ndcg)
    rla.add_metric(precision)
    rla.add_metric(recall)
    results = rla.compute(recs, test)


    return results.mean()

print(itemknn_lenskit_ml100k())