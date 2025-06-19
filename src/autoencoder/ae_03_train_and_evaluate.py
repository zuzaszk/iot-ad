import numpy as np
import json
from ae_01_model import Autoencoder, train_ae, get_reconstruction_errors, evaluate_threshold
from sklearn.metrics import classification_report, confusion_matrix

def final_evaluation(X_train, X_test, y_test, best_params, output_dir="results/ae"):
    model = Autoencoder(input_dim=X_train.shape[1], hidden_dim=best_params["hidden_dim"])
    train_ae(model, X_train, epochs=best_params["epochs"], lr=best_params["lr"], batch_size=best_params["batch_size"])

    test_errors = get_reconstruction_errors(model, X_test)
    threshold = np.percentile(test_errors, 100 * (1 - best_params["contamination"]))
    y_pred = (test_errors > threshold).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    results = {
        "best_params": best_params,
        "threshold": float(threshold),
        "test_classification_report": report,
        "test_confusion_matrix": conf_matrix
    }

    with open(f"{output_dir}/final_test_evaluation.json", "w") as f:
        json.dump(results, f, indent=4)

    print("AE Final Test Evaluation Saved.")

if __name__ == "__main__":
    X_train = np.load("data/processed/baseline/X_train.npy")
    X_test = np.load("data/processed/baseline/X_test.npy")
    y_test = np.load("data/processed/baseline/y_test.npy")

    with open("results/ae/best_params.json", "r") as f:
        best_params = json.load(f)

    final_evaluation(X_train, X_test, y_test, best_params)
