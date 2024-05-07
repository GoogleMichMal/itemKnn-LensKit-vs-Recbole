from recbole.quick_start import run_recbole
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