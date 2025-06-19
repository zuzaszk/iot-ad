import numpy as np
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from tqdm import tqdm
import json
from ae_01_model import Autoencoder, train_ae, get_reconstruction_errors

def cross_validate_autoencoder(X_train, param_grid, output_dir="results/ae"):
    y_dummy = np.zeros(len(X_train))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    best_score = float("inf")
    best_params = None

    for params in tqdm(ParameterGrid(param_grid), desc="AE CV Tuning"):
        fold_errors = []
        for train_idx, val_idx in cv.split(X_train, y_dummy):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            model = Autoencoder(input_dim=X_train.shape[1], hidden_dim=params["hidden_dim"])
            train_ae(model, X_tr, epochs=params["epochs"], lr=params["lr"], batch_size=params["batch_size"], loss_fn=params["loss_fn"])

            val_errors = get_reconstruction_errors(model, X_val)
            threshold = np.percentile(val_errors, 100 * (1 - params["contamination"]))
            fpr = np.mean(val_errors > threshold)
            fold_errors.append(fpr)

        avg_fpr = np.mean(fold_errors)
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
        "hidden_dim": [16, 32, 64],
        "epochs": [20, 50],
        "lr": [1e-3, 1e-4],
        "batch_size": [128, 256],
        "contamination": [0.01, 0.05, 0.1],
        "loss_fn": ["mse", "mae"]
    }

    best_params = cross_validate_autoencoder(X_train, param_grid)
    print("Best AE params:", best_params)
