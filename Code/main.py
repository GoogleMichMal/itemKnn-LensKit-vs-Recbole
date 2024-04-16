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
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.bar(['Lenskit', 'RecBole'], [ndcg_lenskit, ndcg_recbole], color=['blue', 'red'])
plt.title('NDCG@10')
plt.ylim([0, 0.5])
plt.subplot(1, 3, 2)
plt.bar(['Lenskit', 'RecBole'], [precision_lenskit, precision_recbole], color=['blue', 'red'])
plt.title('Precision@10')
plt.ylim([0, 0.5]) 
plt.subplot(1, 3, 3)
plt.bar(['Lenskit', 'RecBole'], [recall_lenskit, recall_recbole], color=['blue', 'red'])
plt.title('Recall@10')
plt.ylim([0, 0.5])
plt.show()
