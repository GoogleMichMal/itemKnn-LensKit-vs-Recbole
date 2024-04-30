from itemknnlenskit import itemknn_lenskit_ml100k
from itemknnrecbole import itemknn_recbole_ml100k
import matplotlib.pyplot as plt
import seaborn as sns

result_lenskit_ml100k = itemknn_lenskit_ml100k()
result_recbole = itemknn_recbole_ml100k()

ndcg_lenskit_ml100k = result_lenskit_ml100k.ndcg
precision_lenskit_ml100k = result_lenskit_ml100k.precision
recall_lenskit_ml100k = result_lenskit_ml100k.recall

ndcg_recbole_ml100k = result_recbole['ndcg@10']
precision_recbole_ml100k = result_recbole['precision@10']
recall_recbole_ml100k = result_recbole['recall@10']

models = ["Lenskit", "RecBole"]
ndcg_scores_ml100k = [ndcg_lenskit_ml100k, ndcg_recbole_ml100k]
precision_scores_ml100k = [precision_lenskit_ml100k, precision_recbole_ml100k]
recall_scores_ml100k = [recall_lenskit_ml100k, recall_recbole_ml100k]


# Set seaborn style
sns.set_style("whitegrid")

# Create figure and axes
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot NDCG@10
sns.barplot(x=models, y=ndcg_scores_ml100k, palette=["#4c72b0", "#55a868"], ax=axs[0], hue=models, legend=False)
axs[0].set_title("NDCG@10")
axs[0].set_ylim([0, 0.5])
for i, score in enumerate(ndcg_scores_ml100k):
    axs[0].text(i, score + 0.02, f"{score:.2f}", ha="center", va="bottom")

# Plot Precision@10
sns.barplot(x=models, y=precision_scores_ml100k, palette=["#4c72b0", "#55a868"], ax=axs[1], hue=models, legend=False)
axs[1].set_title("Precision@10")
axs[1].set_ylim([0, 0.5])
for i, score in enumerate(precision_scores_ml100k):
    axs[1].text(i, score + 0.02, f"{score:.2f}", ha="center", va="bottom")

# Plot Recall@10
sns.barplot(x=models, y=recall_scores_ml100k, palette=["#4c72b0", "#55a868"], ax=axs[2], hue=models, legend=False)
axs[2].set_title("Recall@10")
axs[2].set_ylim([0, 0.5])
for i, score in enumerate(recall_scores_ml100k):
    axs[2].text(i, score + 0.02, f"{score:.2f}", ha="center", va="bottom")

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()
