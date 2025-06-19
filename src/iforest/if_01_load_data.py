import numpy as np

def baseline_load_data():
    X_train = np.load("data/processed/baseline/X_train.npy")
    X_test = np.load("data/processed/baseline/X_test.npy")
    y_test = np.load("data/processed/baseline/y_test.npy")
    return X_train, X_test, y_test