from lenskit.crossfold import partition_users, SampleFrac


def lk_partition_users(data, seed=42):
    # We use LensKit built-in user partition to perform user partitioning
    return next(
        partition_users(
            data[["user", "item", "rating"]],
            1,
            SampleFrac(0.2, rng_spec=seed),
            rng_spec=seed,
        )
    )
