### This directory contains our "replication" results.
Here's what we did:

1. **Dataset and Algorithms**: We ran both the RecBole and LensKit itemKNN algorithms for an initial comparison.
2. **Metric Comparison**: We compared the nDCG@10, Precision, and Recall metrics between the two algorithms.
3. **Discrepancy Noticed**: Noticed significant differences in results, motivating further investigation into the algorithms' implementations.

The files listed in this directory contain the results of our initial comparisons. The code used for these experiments can be found in the "Code" directory. If you are interested in replicating our results, run the files contained in this directory.

### Files in this Directory:
- `lk_partition_users.py`
- `lk_replication.py`
- `recbole_replication.py`
- `replication_plot.py`

### How to Use:
- **RecBole Replication**: Run `recbole_replication.py` to perform the replication using RecBole.
- **LensKit Replication**: Run `lk_replication.py` and `lk_partition_users.py` to perform the replication using LensKit.
- **Plot Results**: Use `replication_plot.py` to visualize the results of the replication.

For detailed instructions on running the code, please refer to the comments within each script.
