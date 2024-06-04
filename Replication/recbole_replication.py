from recbole.quick_start import run_recbole
def itemknn_recbole():
    result = run_recbole(model='ItemKNN', dataset='ml-100k', config_file_list=['Replication/replication.yaml'])
    return result['test_result']

itemknn_recbole()