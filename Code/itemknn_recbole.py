import sys
import torch.distributed as dist
import pandas as pd

from recbole.model.general_recommender.itemknn import ItemKNN
from recbole.evaluator import Collector
from recbole.trainer import TraditionalTrainer
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from logging import getLogger
from nDCG import nDCG_RB
from recbole.utils import (
    init_logger,
    init_seed,
    get_environment,
)


"""
itemknn_recbole
########################


This script is used to calculate the nDCG for the ItemKNN algorithm on the datasets: ml100k, ml1m, anime, modcloth.
The datasets that are used have been splitted by RecBole in order to make sure, that both frameworks use exactly the same (splitted) 
data.
"""



    

def runitemknn_recbole(dataset="ml-100k"):
    """Run the ItemKNN algorithm on the RecBole framework and calculate the nDCG.

    Args:
        dataset (str, optional): The dataset that should be used. Defaults to "ml-100k".
    """

    config = Config(model='ItemKNN', dataset='ml-100k', config_file_list=['Data/ml-100k/recbole.yaml'])
    if(dataset == "ml-100k"):
        pass
    elif(dataset == "book-crossing"):
        config = Config(model='ItemKNN', dataset='book-crossing', config_file_list=['Data/book-crossing/recbole_bookcrossing.yaml'])
    elif(dataset == "ml-1m"):
        config = Config(model='ItemKNN', dataset='ml-1m', config_file_list=['Data/ml-1m/recbole_ml1m.yaml'])
    elif(dataset == "anime"):
        config = Config(model='ItemKNN', dataset='anime', config_file_list=['Data/anime/recbole_anime.yaml'])
    elif(dataset == "modcloth"):
        config = Config(model='ItemKNN', dataset='modcloth', config_file_list=['Data/modcloth/recbole_modcloth.yaml'])


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
    ndcg = nDCG_RB(config).calculate(struct)

    #log environment information
    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
        )

    if not config["single_spec"]:
        dist.destroy_process_group()

    return ndcg



def toDataframe(test, train):
    """Convert a RecBole 'Dataset' object to a pandas DataFrame (used in order to make sure that LensKit and RecBole use the same data)

    Args:
        test (Dataset): RecBole test dataset
        train (Dataset): RecBole train dataset

    Returns:
        df1 (DataFrame): test dataset as pandas DataFrame
        df2 (DataFrame): train dataset as pandas DataFrame
    """
    testFrame = {'user': test['user_id'], 'item': test['item_id'], 'rating': test['label']}
    df1 = pd.DataFrame(data=testFrame)

    trainFrame = {'user': train['user_id'], 'item': train['item_id'], 'rating': train['label']}
    df2 = pd.DataFrame(data=trainFrame)
    return df1, df2

print(runitemknn_recbole("ml-100k"))