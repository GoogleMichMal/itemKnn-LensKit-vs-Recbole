from recbole.quick_start import run_recbole
def itemknn_recbole_ml100k():
    result = run_recbole(model='ItemKNN', dataset='ml-100k', config_file_list=['Code/recbole.yaml'])
    return result['test_result']
