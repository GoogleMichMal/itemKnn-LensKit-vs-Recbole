import seaborn as sns
import matplotlib.pyplot as plt


###### ML-100k ######
# Data
ndcg_lenskit_ml100k = 0.2676
ndcg_recbole_ml100k = 0.2674
models_ml100k = ["Lenskit", "RecBole"]
ndcg_scores_ml100k = [ndcg_lenskit_ml100k, ndcg_recbole_ml100k]

# Plot
sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=models_ml100k, y=ndcg_scores_ml100k, palette="viridis")

for i in range(len(ndcg_scores_ml100k)):
    ax.text(i, ndcg_scores_ml100k[i] + 0.01, f'{ndcg_scores_ml100k[i]:.4f}', ha='center', va='bottom')

plt.xlabel('Models')
plt.ylabel('NDCG Scores')
plt.title('NDCG@10 - ML100K')
plt.ylim(0, 0.5)
plt.show()


###### ML-1M ######
# Data
ndcg_lenskit_ml1m = 0.2586
ndcg_recbole_ml1m = 0.2586
models_ml1m = ["Lenskit", "RecBole"]
ndcg_scores_ml1m = [ndcg_lenskit_ml1m, ndcg_recbole_ml1m]

# Plot
sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=models_ml1m, y=ndcg_scores_ml1m, palette="viridis")

for i in range(len(ndcg_scores_ml1m)):
    ax.text(i, ndcg_scores_ml1m[i] + 0.01, f'{ndcg_scores_ml1m[i]:.4f}', ha='center', va='bottom')

plt.xlabel('Models')
plt.ylabel('NDCG Scores')
plt.title('NDCG@10 - ML1M')
plt.ylim(0, 0.5)
plt.show()


###### Anime ######
# Data
ndcg_lenskit_anime = 0.3437
ndcg_recbole_anime = 0.3437
models_anime = ["Lenskit", "RecBole"]
ndcg_scores_anime = [ndcg_lenskit_anime, ndcg_recbole_anime]

# Plot
sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=models_anime, y=ndcg_scores_anime, palette="viridis")

for i in range(len(ndcg_scores_anime)):
    ax.text(i, ndcg_scores_anime[i] + 0.01, f'{ndcg_scores_anime[i]:.4f}', ha='center', va='bottom')

plt.xlabel('Models')
plt.ylabel('NDCG Scores')
plt.title('NDCG@10 - Anime')
plt.ylim(0, 0.5)
plt.show()


###### ModCloth ######
# Data
ndcg_lenskit_modcloth = 0.0927
ndcg_recbole_modcloth = 0.0935
models_modcloth = ["Lenskit", "RecBole"]
ndcg_scores_modcloth = [ndcg_lenskit_modcloth, ndcg_recbole_modcloth]

# Plot
sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=models_modcloth, y=ndcg_scores_modcloth, palette="viridis")

for i in range(len(ndcg_scores_modcloth)):
    ax.text(i, ndcg_scores_modcloth[i] + 0.01, f'{ndcg_scores_modcloth[i]:.4f}', ha='center', va='bottom')

plt.xlabel('Models')
plt.ylabel('NDCG Scores')
plt.title('NDCG@10 - modcloth')
plt.ylim(0, 0.5)
plt.show()