import matplotlib.pyplot as plt
import seaborn as sns

# Data for ML100K
ndcg_lenskit_ml100k = 0.2248
precision_lenskit_ml100k = 0.2738
recall_lenskit_ml100k = 0.1828

ndcg_recbole_ml100k = 0.3747
precision_recbole_ml100k = 0.3187
recall_recbole_ml100k = 0.2086

models_ml100k = ["Lenskit", "RecBole"]
ndcg_scores_ml100k = [ndcg_lenskit_ml100k, ndcg_recbole_ml100k]
precision_scores_ml100k = [precision_lenskit_ml100k, precision_recbole_ml100k]
recall_scores_ml100k = [recall_lenskit_ml100k, recall_recbole_ml100k]

# Data for Book Crossing
ndcg_lenskit_bookcrossing = 0.0089
precision_lenskit_bookcrossing = 0.0033
recall_lenskit_bookcrossing = 0.0103

ndcg_recbole_bookcrossing = 0.0239
precision_recbole_bookcrossing = 0.0229
recall_recbole_bookcrossing = 0.0091

models_bookcrossing = ["Lenskit", "RecBole"]
ndcg_scores_bookcrossing = [ndcg_lenskit_bookcrossing, ndcg_recbole_bookcrossing]
precision_scores_bookcrossing = [
    precision_lenskit_bookcrossing,
    precision_recbole_bookcrossing,
]
recall_scores_bookcrossing = [recall_lenskit_bookcrossing, recall_recbole_bookcrossing]

# Set seaborn style
sns.set_style("whitegrid")

# Create subplots for ML100K
fig_ml100k, axs_ml100k = plt.subplots(1, 3, figsize=(15, 5))

# Plot NDCG@10 for ML100K
sns.barplot(
    x=models_ml100k,
    y=ndcg_scores_ml100k,
    palette=["#4c72b0", "#55a868"],
    ax=axs_ml100k[0],
    hue=models_ml100k,
    legend=False,
)
axs_ml100k[0].set_title("NDCG@10 - ML100K")
axs_ml100k[0].set_ylim([0, 0.5])

# Plot Precision@10 for ML100K
sns.barplot(
    x=models_ml100k,
    y=precision_scores_ml100k,
    palette=["#4c72b0", "#55a868"],
    ax=axs_ml100k[1],
    hue=models_ml100k,
    legend=False,
)
axs_ml100k[1].set_title("Precision@10 - ML100K")
axs_ml100k[1].set_ylim([0, 0.5])

# Plot Recall@10 for ML100K
sns.barplot(
    x=models_ml100k,
    y=recall_scores_ml100k,
    palette=["#4c72b0", "#55a868"],
    ax=axs_ml100k[2],
    hue=models_ml100k,
    legend=False,
)
axs_ml100k[2].set_title("Recall@10 - ML100K")
axs_ml100k[2].set_ylim([0, 0.5])

# Annotate bars with scores
for ax in axs_ml100k:
    for p in ax.patches:
        ax.annotate(
            format(p.get_height(), ".4f"),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
        )

# Adjust layout and display plot
plt.tight_layout()
plt.show()

# Create subplots for Book Crossing
fig_bookcrossing, axs_bookcrossing = plt.subplots(1, 3, figsize=(15, 5))

# Plot NDCG@10 for Book Crossing
sns.barplot(
    x=models_bookcrossing,
    y=ndcg_scores_bookcrossing,
    palette=["#4c72b0", "#55a868"],
    ax=axs_bookcrossing[0],
    hue=models_bookcrossing,
    legend=False,
)
axs_bookcrossing[0].set_title("NDCG@10 - Book Crossing")
axs_bookcrossing[0].set_ylim([0, 0.05])

# Plot Precision@10 for Book Crossing
sns.barplot(
    x=models_bookcrossing,
    y=precision_scores_bookcrossing,
    palette=["#4c72b0", "#55a868"],
    ax=axs_bookcrossing[1],
    hue=models_bookcrossing,
    legend=False,
)
axs_bookcrossing[1].set_title("Precision@10 - Book Crossing")
axs_bookcrossing[1].set_ylim([0, 0.05])

# Plot Recall@10 for Book Crossing
sns.barplot(
    x=models_bookcrossing,
    y=recall_scores_bookcrossing,
    palette=["#4c72b0", "#55a868"],
    ax=axs_bookcrossing[2],
    hue=models_bookcrossing,
    legend=False,
)
axs_bookcrossing[2].set_title("Recall@10 - Book Crossing")
axs_bookcrossing[2].set_ylim([0, 0.05])

# Annotate bars with scores
for ax in axs_bookcrossing:
    for p in ax.patches:
        ax.annotate(
            format(p.get_height(), ".4f"),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
        )

# Adjust layout and display plot
plt.tight_layout()
plt.show()
