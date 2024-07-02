### This directory contains our "reproduction" results.
Here's what we did:

1. **Adjusted nDCG Calculation**: We modified the nDCG@10 calculation for LensKit to match the RecBole implementation.
2. **Data Splitting**: Using RecBole, we split the data into train/test sets with an 80/20 ratio, converted these sets into CSV files, and utilized them for running the LensKit algorithm, ensuring identical datasets for both algorithms.
3. **Similarity Matrix Adjustment**: We aligned the LensKit similarity matrix calculation to correspond with the RecBole implementation. Details of this adjustment are provided in the file `split_itemknn_modified`.

The files listed in this directory do not contain our code. The code can be found in the "Code" directory. If you are interested in plotting our results, run the files contained in this directory.

### Files in this Directory:
- `plot_itemknn_modified.py`
- `plot_sameData_sameNDCG.py`

How to Use:
- **Plot Results**: Use `plot_itemknn_modified.py` to visualize the results of the adjusted LensKit calculation.
- **Plot Same Data, Same nDCG**: Use `plot_sameData_sameNDCG.py` to visualize the results of the same data and same nDCG calculation.

For detailed instructions on running the code, please refer to the comments within each script.
