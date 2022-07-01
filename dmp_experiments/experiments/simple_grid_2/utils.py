from dmp.data.pmlb import pmlb_loader
from sklearn.model_selection import train_test_split
import json
import numpy as np

def load_data():
    datasets = pmlb_loader.load_dataset_index()
    dataset, X, y = pmlb_loader.load_dataset(datasets, '537_houses')

    # any transformations?

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    return X_train, X_test, y_train, y_test


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)