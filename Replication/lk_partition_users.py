from lenskit.crossfold import partition_users, SampleFrac

def lk_partition_users(data):
    # we use lenskit built-in user partition to perform user partitioning
    return next(partition_users(data[['user', 'item', 'rating']], 1, SampleFrac(0.2, rng_spec=42), rng_spec=42))

