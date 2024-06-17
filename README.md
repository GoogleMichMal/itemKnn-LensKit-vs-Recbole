# ItemKnn-LensKit-vs-Recbole

This repository aims to study, highlight, and understand the differences in the item-based kNN implementation between the libraries [LensKit](https://lkpy.readthedocs.io/en/stable/knn.html) and [Recbole](https://recbole.io/docs/user_guide/model/general/itemknn.html).

### Data Sets

We utilized four datasets for our experiments: Anime, Modcloth, ML-100K, and ML-1M. These datasets were sourced from the Google Drive folder provided by RecBole. Each dataset was pre-processed into implicit feedback format suitable for recommendation systems.

### Algorithms

We focused on comparing the performance of the item-based kNN algorithm between LensKit and RecBole. Both implementations were configured with a k value of 20 and tasked to generate 10 recommendations per user in the test dataset.

## Setup

To replicate our experiments, follow these steps:

1. **Clone the Repository**: 
```
git clone https://github.com/GoogleMichMal/itemKnn-LensKit-vs-Recbole.git
```

2. **Download the Data**:
Download the [Data.zip](https://github.com/GoogleMichMal/itemKnn-LensKit-vs-Recbole/releases/latest/download/Data.zip) file or navigate to the `Code` directory and run the `download-data.py` script to automatically download and extract the datasets required for the experiments. It contains all datasets used in our experiments.


3. **Unpack Data**:
Unpack Data.zip in the parent directory of the repository, such that "Code" and "Data" are in the same directory.

4. **Install Requirements**:
In the parent directory, run:
```
pip install -r requirements.txt
```


