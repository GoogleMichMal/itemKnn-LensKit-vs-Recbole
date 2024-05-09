from recbole.quick_start import run_recbole
from recbole.config import Config
from recbole.data.transform import construct_transform
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender.itemknn import ItemKNN
from recbole.utils import init_seed, get_model, get_trainer

def runitemknn_recbole(config):
    dataset = create_dataset(config['model'])
    print(dataset)
    #data splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    #model loading
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    transform = construct_transform(config)

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=True, show_progress=config["show_progress"]
    )
    
    result = {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }
    return result


cfg = Config(model='ItemKNN', dataset='ml-100k', config_file_list=['Data/ml-100k/recbole.yaml'])
print(runitemknn_recbole(cfg))


def itemknn_recbole_ml100k():
    result = run_recbole(model='ItemKNN', dataset='ml-100k', config_file_list=['Data/ml-100k/recbole.yaml'])
    return result['test_result']

def itemknn_recbole_bookcrossing():
    result = run_recbole(model='ItemKNN', dataset='book-crossing', config_file_list=['Data/book-crossing/recbole_bookcrossing.yaml'])
    return result['test_result']

def itemknn_recbole_amazon():
    result = run_recbole(model='ItemKNN', dataset='Amazon_CDs_and_Vinyl', config_file_list=['Data/Amazon_CDs_and_Vinyl/recbole_amazon_cds_and_vinyl.yaml'])
    return result['test_result']

def itemknn_recbole_ml1m():
    result = run_recbole(model='ItemKNN', dataset='ml-1m', config_file_list=['Data/ml-1m/recbole_ml1m.yaml'])
    return result['test_result']

def itemknn_recbole_food():
    result = run_recbole(model='ItemKNN', dataset='Food', config_file_list=['Data/Food/recbole_food.yaml'])
    return result['test_result']

def itemknn_recbole_ml20m():
    result = run_recbole(model='ItemKNN', dataset='ml-20m', config_file_list=['Data/ml-20m/recbole_ml20m.yaml'])
    return result['test_result']