import pandas as pd
from lenskit.datasets import ML100K
from lenskit import batch, topn
from lenskit.algorithms import Recommender, item_knn


def itemknn_lenskit():
    ml100k = ML100K("data/ml-100k")
    ratings = ml100k.ratings
    ratings.head()

    test = ratings.sample(frac=0.2, random_state=42)
    train = ratings.drop(test.index)
    print("Train:", train.shape, "Test:", test.shape)

    item_knn = item_knn.ItemItem(20, feedback="implicit")

    fittable = Recommender.adapt(item_knn)

    fittable.fit(train)

    users = test.user.unique()

    recs = batch.recommend(fittable, users, 10, n_jobs=1)

    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    results = rla.compute(recs, test)
    print(f"NDCG@10: {results.mean()}")
