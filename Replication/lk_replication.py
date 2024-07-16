import pandas as pd
from lenskit import batch, topn
from lenskit.algorithms import Recommender, item_knn
from lenskit.metrics.topn import ndcg, precision, recall
from lk_partition_users import lk_partition_users

def itemknn_lenskit():
    ml100k = pd.read_csv("Data/ml-100k/u.data", header=None, sep="\t", names=["user", "item", "rating", "timestamp"])

    train, test = lk_partition_users(ml100k)

    itemknn = item_knn.ItemItem(20, feedback="implicit")

    fittable = Recommender.adapt(itemknn)

    fittable.fit(train)

    users = test.user.unique()

    recs = batch.recommend(fittable, users, 10, n_jobs=1)

    rla = topn.RecListAnalysis()
    rla.add_metric(ndcg, name="ndcg", k=10)
    rla.add_metric(precision, name="precision", k=10)
    rla.add_metric(recall, name="recall", k=10)
    results = rla.compute(recs, test)

    return results.mean()

print(itemknn_lenskit())