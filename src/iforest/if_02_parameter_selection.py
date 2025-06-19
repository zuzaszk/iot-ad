from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold, ParameterGrid, cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np
import json

def parameter_selection(X_train, param_grid, output_dir="results/iforest"):

    best_model = None
    best_score = float("inf")
    results_summary = []

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_dummy = np.zeros(len(X_train))

    for params in tqdm(ParameterGrid(param_grid), desc="Hyperparameter Tuning"):
        model = IsolationForest(**params)
        y_pred_cv = cross_val_predict(model, X_train, y_dummy, cv=cv, method='predict')
        y_pred = np.where(y_pred_cv == -1, 1, 0)

        fpr = np.sum(y_pred) / len(y_pred)

        result = {
            "params": params,
            "false_positives": int(np.sum(y_pred)),
            "false_positive_rate": fpr,
            "precision": precision_score(y_dummy, y_pred, zero_division=0),
            "recall": recall_score(y_dummy, y_pred, zero_division=0),
            "f1_score": f1_score(y_dummy, y_pred, zero_division=0)
        }
        results_summary.append(result)

        if fpr < best_score:
            best_score = fpr
            best_model = model

    with open(f"{output_dir}/all_cv_results.json", "w") as f:
        json.dump(results_summary, f, indent=4)

    return best_model, results_summary
