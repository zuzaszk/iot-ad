import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from if_01_load_data import baseline_load_data
from if_02_parameter_selection import parameter_selection

param_grid = {
    "n_estimators": [50, 100, 200, 500],
    "max_samples": [0.3, 0.5, 0.7, 1.0],
    "contamination": [0.01, 0.05, 0.1],
    "max_features": [1.0, 0.5, 0.3, 0.1],
    "random_state": [42, 123],
    "bootstrap": [True, False],
    "n_jobs": [-1]
}

output_dir = "results/iforest2"
X_train, X_test, y_test = baseline_load_data()
best_model, _ = parameter_selection(X_train, param_grid, output_dir=output_dir)

best_model.fit(X_train)
y_pred_test = best_model.predict(X_test)
y_pred_test_binary = np.where(y_pred_test == -1, 1, 0)

report = classification_report(y_test, y_pred_test_binary, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred_test_binary).tolist()

final_results = {
    "best_params": best_model.get_params(),
    "test_classification_report": report,
    "test_confusion_matrix": conf_matrix
}

with open(f"{output_dir}/final_test_evaluation.json", "w") as f:
    json.dump(final_results, f, indent=4)

print("Final test evaluation saved.")
