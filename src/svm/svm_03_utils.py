import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score

def train_ocsvm(X_train, params):
    model = OneClassSVM(**params)
    model.fit(X_train)
    return model

def predict_and_evaluate(model, X, threshold=None):
    preds_raw = model.predict(X)
    preds = np.where(preds_raw == -1, 1, 0)
    return preds
