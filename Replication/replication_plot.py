import matplotlib.pyplot as plt
import seaborn as sns

## This file plots the nDCG@10, Precision@10 and Recall@10 results for the Lenskit and RecBole algorithms.

ndcg_lenskit = 0.3067
precision_lenskit = 0.2749
recall_lenskit = 0.3206

ndcg_recbole = 0.3747
precision_recbole = 0.3187
recall_recbole = 0.2086

models = ["Lenskit", "RecBole"]
ndcg_scores = [ndcg_lenskit, ndcg_recbole]
precision_scores = [precision_lenskit, precision_recbole]
recall_scores = [recall_lenskit, recall_recbole]


# Set seaborn style
sns.set_style("whitegrid")

# Create figure and axes
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot NDCG@10
sns.barplot(x=models, y=ndcg_scores, palette=["#4c72b0", "#55a868"], ax=axs[0], hue=models, legend=False)
axs[0].set_title("NDCG@10")
axs[0].set_ylim([0, 0.5])
for i, score in enumerate(ndcg_scores):
    axs[0].text(i, score + 0.02, f"{score:.4f}", ha="center", va="bottom")

# Plot Precision@10
sns.barplot(x=models, y=precision_scores, palette=["#4c72b0", "#55a868"], ax=axs[1], hue=models, legend=False)
axs[1].set_title("Precision@10")
axs[1].set_ylim([0, 0.5])
for i, score in enumerate(precision_scores):
    axs[1].text(i, score + 0.02, f"{score:.4f}", ha="center", va="bottom")

# Plot Recall@10
sns.barplot(x=models, y=recall_scores, palette=["#4c72b0", "#55a868"], ax=axs[2], hue=models, legend=False)
axs[2].set_title("Recall@10")
axs[2].set_ylim([0, 0.5])
for i, score in enumerate(recall_scores):
    axs[2].text(i, score + 0.02, f"{score:.4f}", ha="center", va="bottom")

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()