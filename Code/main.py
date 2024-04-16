from itemknnlenskit import itemknn_lenskit
from itemknnrecbole import itemknn_recbole
import matplotlib.pyplot as plt

result_lenskit = itemknn_lenskit()
result_recbole = itemknn_recbole()

ndcg_lenskit = result_lenskit.ndcg
precision_lenskit = result_lenskit.precision
recall_lenskit = result_lenskit.recall

ndcg_recbole = result_recbole['ndcg@10']
precision_recbole = result_recbole['precision@10']
recall_recbole = result_recbole['recall@10']


# Plotting the results of the two algorithms side by side for ndcg, precision and recall



