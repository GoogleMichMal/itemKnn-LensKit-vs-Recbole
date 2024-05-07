import pandas as pd
from lenskit.datasets import ML100K
from lenskit import batch, topn
from lenskit.algorithms import Recommender, item_knn
from tqdm import tqdm


def itemknn_lenskit_ml100k():
    # Read the MovieLens 100K dataset
    ml100k = ML100K("data/ml-100k")
    ratings = ml100k.ratings
    ratings.head()

    # Split the data into training and test sets
    test = ratings.sample(frac=0.2, random_state=42)
    train = ratings.drop(test.index)

    itemknn = item_knn.ItemItem(20, feedback="implicit")

    fittable = Recommender.adapt(itemknn)

    fittable.fit(train)

    users = test.user.unique()

    recs = batch.recommend(fittable, users, 10, n_jobs=1)

    # Initialize the RecListAnalysis object
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    rla.add_metric(topn.precision)
    rla.add_metric(topn.recall)
    results = rla.compute(recs, test)
    return results.mean()

def itemknn_lenskit_bookcrossing():
    # Read the Book-Crossing dataset
    bookcrossing = pd.read_csv("Data/book-crossing/Ratings.csv", sep=";", skiprows=1, names=["user", "item", "rating"], dtype={"user": str, "item": str, "rating": float})

    # Print out the number of unique users and items
    num_unique_users = bookcrossing['user'].nunique()
    num_unique_items = bookcrossing['item'].nunique()
    print("Number of unique users:", num_unique_users)
    print("Number of unique items:", num_unique_items)


    # Split the data into training and test sets
    test = bookcrossing.sample(frac=0.2, random_state=42)
    train = bookcrossing.drop(test.index)

    # Create an ItemItem recommender
    itemknn = item_knn.ItemItem(20, feedback="implicit")

    # Adapt the recommender to the LensKit API
    fittable = Recommender.adapt(itemknn)
    fittable.fit(train)

    # Get the unique users
    users = list(test.user.unique())  

    recs = []
    for user in tqdm(users, desc="Generating Recommendations"):
        recs.append(batch.recommend(fittable, [user], 10, n_jobs=1).assign(user=user))

    # Combine the recommendations into a single DataFrame
    recs_df = pd.concat(recs, ignore_index=True)  

    # Initialize the RecListAnalysis object
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    rla.add_metric(topn.precision)
    rla.add_metric(topn.recall)

    # Compute the evaluation metrics
    results = rla.compute(recs_df, test)
    return results.mean()
    

def itemknn_lenskit_food():
    food = pd.read_csv("Data/food/RAW_interactions.csv", sep=",", quotechar='"', skiprows=1, usecols=[0,1,3], names=["user", "item", "rating"], dtype={"user": str, "item": str, "rating": float})
    
    num_unique_users = food['user'].nunique()
    num_unique_items = food['item'].nunique()
    print("Number of unique users:", num_unique_users)
    print("Number of unique items:", num_unique_items)

    test = food.sample(frac=0.2, random_state=42)
    train = food.drop(test.index)

    itemknn = item_knn.ItemItem(20, feedback="implicit")

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
