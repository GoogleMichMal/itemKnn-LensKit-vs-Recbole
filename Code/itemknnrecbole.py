import sys
import torch.distributed as dist
import numpy as np
import pandas as pd

from recbole.evaluator.base_metric import TopkMetric
from recbole.model.general_recommender.itemknn import ItemKNN
from recbole.evaluator import Collector
from recbole.trainer import TraditionalTrainer
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from logging import getLogger
from recbole.utils import (
    init_logger,
    init_seed,
    get_environment,
)

class ndcgRecbole(TopkMetric):
    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        # pos_index: a bool matrix, shape user * k. The item with the (j+1)-th highest score of i-th user is pos if pos_index[i][j] is True
        # pos_len: a vector representing the number of positive items for each user
        pos_index, pos_len = self.used_info(dataobject)
        result = self.metric_info(pos_index, pos_len)
        metric_dict = self.topk_result("ndcg", result)
        return metric_dict
    
    # recbole-ndcg implementation
    def metric_info(self, pos_index, pos_len):
        # len_rank: [10, 10, 10, 10, ... userNumber]
        len_rank = np.full_like(pos_len, pos_index.shape[1])
        idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)
        
        # iranks:  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ...]
        iranks = np.zeros_like(pos_index, dtype=np.float)
        # iranks:  [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ...]
        iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
        idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
        for row, idx in enumerate(idcg_len):
            # for user with less than 10 positive items, fill the rest with the last valid value
            idcg[row, idx:] = idcg[row, idx - 1]

        # ranks:  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ...]
        ranks = np.zeros_like(pos_index, dtype=np.float)
        # ranks:  [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ...]
        ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)

        dcg = 1.0 / np.log2(ranks + 1)
        dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)
        result = dcg / idcg
        return result
    

# Config object: config = (model, dataset, config_file_list, config_dict)
def runitemknn_recbole(dataset="ml-100k"):
    config = Config(model='ItemKNN', dataset='ml-100k', config_file_list=['Data/ml-100k/recbole.yaml'])
    if(dataset == "ml-100k"):
        pass
    elif(dataset == "book-crossing"):
        config = Config(model='ItemKNN', dataset='book-crossing', config_file_list=['Data/book-crossing/recbole_bookcrossing.yaml'])
    elif(dataset == "ml-1m"):
        config = Config(model='ItemKNN', dataset='ml-1m', config_file_list=['Data/ml-1m/recbole_ml1m.yaml'])
    elif(dataset == "ml-20m"):
        config = Config(model='ItemKNN', dataset='ml-20m', config_file_list=['Data/ml-20m/recbole_ml20m.yaml'])
    elif(dataset == "anime"):
        config = Config(model='ItemKNN', dataset='anime', config_file_list=['Data/anime/recbole_anime.yaml'])


    # set seed-configuration for math-libraries and random number generators
    init_seed(config["seed"], config["reproducibility"])

    # logger init
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # create Dataset Object 
    # (https://github.com/RUCAIBox/RecBole/blob/master/recbole/data/utils.py)
    dataset = create_dataset(config)
    logger.info(dataset)

    # train/test Split (returns AbstractDataLoader objects)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    print("Train Data: ", train_data._dataset)
    print("Test Data: ", test_data._dataset)

    # dataframe1, dataframe2 = toDataframe(train_data._dataset, test_data._dataset)
    # print("Dataframe1: ", dataframe1)

    # get model object
    model = ItemKNN(config, train_data._dataset).to(config["device"])

    # logging model info
    logger.info(model)

    # get trainer object
    trainer = TraditionalTrainer(config, model)

    # model training (returns best valid score and best valid result. If valid_data is None, it returns (-1, None)
    # (https://github.com/RUCAIBox/RecBole/blob/master/recbole/trainer/trainer.py)
    trainer.fit(train_data, valid_data, saved=True, show_progress=False)

    # evaluation collector
    # (https://github.com/RUCAIBox/RecBole/blob/master/recbole/evaluator/collector.py)
    collector = Collector(config)
    trainer.tot_item_num = dataset.item_num

    
    for _, batched_data in enumerate(test_data):
        interaction, scores, positive_u, positive_i = trainer._full_sort_batch_eval(batched_data)
        collector.eval_batch_collect(scores, interaction, positive_u, positive_i) 
    
    # get data struct
    # (https://github.com/RUCAIBox/RecBole/blob/master/recbole/evaluator/collector.py)
    struct = collector.get_data_struct()
    
    # own ndcg
    ndcg = ndcgRecbole(config)
    result = ndcg.calculate_metric(struct)
    print(result)


    #log environment information
    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
        )

    if not config["single_spec"]:
        dist.destroy_process_group()

    return result



def toDataframe(test, train):
    testFrame = {'user': test['user_id'], 'item': test['item_id'], 'rating': test['rating']}
    df1 = pd.DataFrame(data=testFrame)

    trainFrame = {'user': train['user_id'], 'item': train['item_id'], 'rating': train['rating']}
    df2 = pd.DataFrame(data=trainFrame)
    return df1, df2

print(runitemknn_recbole("ml-1m"))