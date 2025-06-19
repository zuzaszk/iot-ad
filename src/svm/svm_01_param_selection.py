import numpy as np
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from tqdm import tqdm
import json
from pathlib import Path
from svm_03_utils import train_ocsvm, predict_and_evaluate

def cross_validate_ocsvm(X_train, param_grid, output_dir="results/ocsvm"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    y_dummy = np.zeros(len(X_train))  # all benign for unsupervised
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    best_score = float("inf")
    best_params = None

    for params in tqdm(ParameterGrid(param_grid), desc="SVM CV Tuning"):
        fold_fprs = []
        for train_idx, val_idx in cv.split(X_train, y_dummy):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            model = train_ocsvm(X_tr, params)
            preds_val = predict_and_evaluate(model, X_val)

            fpr = np.mean(preds_val == 1)
            fold_fprs.append(fpr)

        avg_fpr = np.mean(fold_fprs)
        results.append({
            "params": params,
            "avg_false_positive_rate": avg_fpr
        })

        if avg_fpr < best_score:
            best_score = avg_fpr
            best_params = params

    with open(f"{output_dir}/all_cv_results.json", "w") as f:
        json.dump(results, f, indent=4)

    with open(f"{output_dir}/best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    return best_params

if __name__ == "__main__":
    X_train = np.load("data/processed/baseline/X_train.npy")

    param_grid = {
        "kernel": ["rbf", "sigmoid"],
        "nu": [0.01, 0.05, 0.1],
        "gamma": ["scale", "auto"]
    }

    best_params = cross_validate_ocsvm(X_train, param_grid)
    print("Best One-Class SVM params:", best_params)
