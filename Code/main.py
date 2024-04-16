from itemknnlenskit import itemknn_lenskit
from itemknnrecbole import itemknn_recbole
import matplotlib.pyplot as plt

result_lenskit = itemknn_lenskit()
result_recbole = itemknn_recbole()
print(result_lenskit)
print(result_recbole)


# Plotting the results of the two algorithms side by side for ndcg, precision and recall



