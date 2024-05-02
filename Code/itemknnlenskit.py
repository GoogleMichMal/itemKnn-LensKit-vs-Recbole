import pandas as pd
from lenskit.datasets import ML100K
from lenskit import batch, topn
from lenskit.algorithms import Recommender, item_knn


def itemknn_lenskit_ml100k():
    ml100k = ML100K("data/ml-100k")
    ratings = ml100k.ratings
    ratings.head()

    test = ratings.sample(frac=0.2, random_state=42)
    train = ratings.drop(test.index)
    print("Train:", train.shape, "Test:", test.shape)

    itemknn = item_knn.ItemItem(20, feedback="implicit")

    fittable = Recommender.adapt(itemknn)

    fittable.fit(train)

    users = test.user.unique()

    recs = batch.recommend(fittable, users, 10, n_jobs=1)

    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    rla.add_metric(topn.precision)
    rla.add_metric(topn.recall)
    results = rla.compute(recs, test)
    return results.mean()

def itemknn_lenskit_bookcrossing():
    bookcrossing = pd.read_csv("/Data/book-crossing/Ratings.csv", sep=";", names=["user", "item", "rating"])

    test = bookcrossing.sample(frac=0.2, random_state=42)
    train = bookcrossing.drop(test.index)
    print("Train:", train.shape, "Test:", test.shape)

    itemknn = item_knn.ItemItem(20, feedback="implicit")

    train["user"] = train["user"].astype(str)

    fittable = Recommender.adapt(itemknn)

    fittable.fit(train)

    users = test.user.unique()

    recs = batch.recommend(fittable, users, 10, n_jobs=1)

    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    rla.add_metric(topn.precision)
    rla.add_metric(topn.recall)
    results = rla.compute(recs, test)
    return results.mean()


print(itemknn_lenskit_bookcrossing())
