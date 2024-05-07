from itemknnlenskit import itemknn_lenskit_ml100k
from itemknnrecbole import itemknn_recbole_ml100k, itemknn_recbole_bookcrossing
import matplotlib.pyplot as plt
import seaborn as sns


# Daten für das MovieLens 100k Dataset
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

# Daten für das Book-Crossing Dataset
ndcg_lenskit_bookcrossing = 0.0089
precision_lenskit_bookcrossing = 0.0033
recall_lenskit_bookcrossing = 0.0103

ndcg_recbole_bookcrossing = 0.0239
precision_recbole_bookcrossing = 0.0229
recall_recbole_bookcrossing = 0.0091

models_bookcrossing = ["Lenskit", "RecBole"]
ndcg_scores_bookcrossing = [ndcg_lenskit_bookcrossing, ndcg_recbole_bookcrossing]
precision_scores_bookcrossing = [precision_lenskit_bookcrossing, precision_recbole_bookcrossing]
recall_scores_bookcrossing = [recall_lenskit_bookcrossing, recall_recbole_bookcrossing]

# Setzen des seaborn-Stils
sns.set_style("whitegrid")

# Erstellen der Plots für das MovieLens 100k Dataset
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# NDCG@10
sns.barplot(x=models_ml100k, y=ndcg_scores_ml100k, palette=["#4c72b0", "#55a868"], ax=axs[0, 0], hue=models_ml100k, legend=False)
axs[0, 0].set_title("NDCG@10 - ML100K")
axs[0, 0].set_ylim([0, 0.5])  # Anpassung der y-Achse

# Precision@10
sns.barplot(x=models_ml100k, y=precision_scores_ml100k, palette=["#4c72b0", "#55a868"], ax=axs[0, 1], hue=models_ml100k, legend=False)
axs[0, 1].set_title("Precision@10 - ML100K")
axs[0, 1].set_ylim([0, 0.5])  # Anpassung der y-Achse

# Recall@10
sns.barplot(x=models_ml100k, y=recall_scores_ml100k, palette=["#4c72b0", "#55a868"], ax=axs[0, 2], hue=models_ml100k, legend=False)
axs[0, 2].set_title("Recall@10 - ML100K")
axs[0, 2].set_ylim([0, 0.5])  # Anpassung der y-Achse

# Erstellen der Plots für das Book-Crossing Dataset
# NDCG@10
sns.barplot(x=models_bookcrossing, y=ndcg_scores_bookcrossing, palette=["#4c72b0", "#55a868"], ax=axs[1, 0], hue=models_bookcrossing, legend=False)
axs[1, 0].set_title("NDCG@10 - Book Crossing")
axs[1, 0].set_ylim([0, 0.05])  # Anpassung der y-Achse

# Precision@10
sns.barplot(x=models_bookcrossing, y=precision_scores_bookcrossing, palette=["#4c72b0", "#55a868"], ax=axs[1, 1], hue=models_bookcrossing, legend=False)
axs[1, 1].set_title("Precision@10 - Book Crossing")
axs[1, 1].set_ylim([0, 0.05])  # Anpassung der y-Achse

# Recall@10
sns.barplot(x=models_bookcrossing, y=recall_scores_bookcrossing, palette=["#4c72b0", "#55a868"], ax=axs[1, 2], hue=models_bookcrossing, legend=False)
axs[1, 2].set_title("Recall@10 - Book Crossing")
axs[1, 2].set_ylim([0, 0.05])  # Anpassung der y-Achse

# Ergebnisse über den Balken anzeigen
for ax_row in axs:
    for ax in ax_row:
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.4f'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 10), 
                        textcoords = 'offset points')

# Layout anpassen
plt.tight_layout()

# Plots anzeigen
plt.show()