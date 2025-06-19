import numpy as np
import json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from svm_03_utils import train_ocsvm, predict_and_evaluate

def final_eval_ocsvm(X_train, X_test, y_test, best_params, output_dir="results/ocsvm"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model = train_ocsvm(X_train, best_params)
    y_pred_test = predict_and_evaluate(model, X_test)

    report = classification_report(y_test, y_pred_test, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred_test).tolist()

    final_results = {
        "best_params": best_params,
        "test_classification_report": report,
        "test_confusion_matrix": conf_matrix
    }

    with open(f"{output_dir}/final_test_evaluation.json", "w") as f:
        json.dump(final_results, f, indent=4)

    print("Final test evaluation saved.")

if __name__ == "__main__":
    X_train = np.load("data/processed/baseline/X_train.npy")
    X_test = np.load("data/processed/baseline/X_test.npy")
    y_test = np.load("data/processed/baseline/y_test.npy")

    with open("results/ocsvm/best_params.json", "r") as f:
        best_params = json.load(f)

    final_eval_ocsvm(X_train, X_test, y_test, best_params)
