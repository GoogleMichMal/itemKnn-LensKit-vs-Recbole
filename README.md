# ItemKnn-LensKit-vs-Recbole

This repository aims to study, highlight, and understand the differences in the item-based kNN implementation between the libraries [LensKit](https://lkpy.readthedocs.io/en/stable/knn.html) and [Recbole](https://recbole.io/docs/user_guide/model/general/itemknn.html).

## Results (So Far)

### MovieLens 100K Dataset

![MovieLens 100K Precision-Recall Curve](https://i.imgur.com/u8hJRPw.png)

### Book-Crossing Dataset

![Book-Crossing Precision-Recall Curve](https://i.imgur.com/StWf7W5.png)

## Setup

To replicate our experiments, follow these steps:

1. **Clone the Repository**: 
```
git clone https://github.com/your_username/ItemKnn-LensKit-vs-Recbole.git
```

2. **Download the Data**:
Download the [Data.zip](https://drive.google.com/uc?export=download&id=1GHBJ-WCp_Y6gPMPfv8h2ZDEyxEXajybe) file. It contains all four datasets used in our experiments.

3. **Unpack Data**:
Unpack Data.zip in the parent directory of the repository, such that "Code" and "Data" are in the same directory.

4. **Install Requirements**:
In the parent directory, run:
```
pip install -r requirements.txt
```
