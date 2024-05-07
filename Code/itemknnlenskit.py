import pandas as pd
from lenskit.datasets import ML100K
from lenskit import batch, topn
from lenskit.algorithms import Recommender, item_knn
from tqdm import tqdm


def itemknn_lenskit_ml100k():
    ml100k = ML100K("data/ml-100k")
    ratings = ml100k.ratings
    ratings.head()

    test = ratings.sample(frac=0.2, random_state=42)
    train = ratings.drop(test.index)

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
    # Lese den gesamten Datensatz ein
    bookcrossing = pd.read_csv("Data/book-crossing/Ratings.csv", sep=";", skiprows=1, names=["user", "item", "rating"], dtype={"user": str, "item": str, "rating": float})

    # Print out the number of unique users and items
    num_unique_users = bookcrossing['user'].nunique()
    num_unique_items = bookcrossing['item'].nunique()
    print("Number of unique users:", num_unique_users)
    print("Number of unique items:", num_unique_items)


    # Teile den Datensatz in Trainings- und Testdaten auf
    test = bookcrossing.sample(frac=0.2, random_state=42)
    train = bookcrossing.drop(test.index)

    # Erstelle den Item-Item Recommender
    itemknn = item_knn.ItemItem(20, feedback="implicit")

    # Passe den Recommender an
    fittable = Recommender.adapt(itemknn)
    fittable.fit(train)

    # Liste aller Benutzer
    users = list(test.user.unique())  

    recs = []
    for user in tqdm(users, desc="Generating Recommendations"):
        recs.append(batch.recommend(fittable, [user], 10, n_jobs=1).assign(user=user))

    # Konvertiere die Empfehlungen in einen DataFrame
    recs_df = pd.concat(recs, ignore_index=True)  

    # Initialisiere das RecListAnalysis-Objekt
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    rla.add_metric(topn.precision)
    rla.add_metric(topn.recall)

    # Berechne die Metriken
    results = rla.compute(recs_df, test)
    return results.mean()
    