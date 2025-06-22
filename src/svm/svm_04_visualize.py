import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Setup paths ===
results_dir = "results/ocsvm"
png_dir = f'{results_dir}/png'
pdf_dir = f'{results_dir}/pdf'
os.makedirs(png_dir, exist_ok=True)
os.makedirs(pdf_dir, exist_ok=True)

# === Load all CV results ===
with open(os.path.join(results_dir, "all_cv_results.json"), "r") as f:
    cv_results = json.load(f)

cv_data = []
for result in cv_results:
    row = result["params"].copy()
    row["false_positive_rate"] = result.get("avg_false_positive_rate", np.nan)
    cv_data.append(row)
cv_df = pd.DataFrame(cv_data)

# === Load final test results ===
with open(os.path.join(results_dir, "final_test_evaluation.json"), "r") as f:
    test_results = json.load(f)

test_report = test_results["test_classification_report"]
conf_matrix = np.array(test_results["test_confusion_matrix"])

# === Figure 1: CV Summary ===
plt.figure(figsize=(10, 6))
sns.boxplot(data=cv_df, x="kernel", y="false_positive_rate", hue="gamma")
plt.title("SVM False Positive Rate by Kernel & Gamma")
plt.ylabel("False Positive Rate")
plt.xlabel("Kernel")
plt.tight_layout()
plt.savefig(os.path.join(png_dir, "svm_cv_boxplot.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(pdf_dir, "svm_cv_boxplot.pdf"), bbox_inches="tight")
plt.close()

# === Figure 2: Nu vs FPR ===
plt.figure(figsize=(8, 6))
sns.lineplot(data=cv_df, x="nu", y="false_positive_rate", marker="o")
plt.title("SVM False Positive Rate by Nu Parameter")
plt.xlabel("Nu")
plt.ylabel("False Positive Rate")
plt.tight_layout()
plt.savefig(os.path.join(png_dir, "svm_nu_fpr.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(pdf_dir, "svm_nu_fpr.pdf"), bbox_inches="tight")
plt.close()

# === Figure 3: Confusion Matrix & Test Metrics ===
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Final SVM Test Results", fontsize=16, fontweight="bold")

# --- Confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt="d",
            xticklabels=["Pred Normal","Pred Anomaly"],
            yticklabels=["True Normal","True Anomaly"],
            cmap="Blues", ax=axes[0, 0])
axes[0, 0].set_title("Confusion Matrix")

# --- Per-class metrics
metrics = ["precision", "recall", "f1-score"]
classes = ["0.0", "1.0"]
vals = np.array([[test_report[c][m] for m in metrics] for c in classes])
x = np.arange(len(metrics))
width = 0.35
axes[0, 1].bar(x - width/2, vals[0], width, label="Normal (0.0)")
axes[0, 1].bar(x + width/2, vals[1], width, label="Anomaly (1.0)")
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(metrics)
axes[0, 1].legend()
axes[0, 1].set_ylim(0, 1)
axes[0, 1].set_title("Classification Metrics by Class")

for i in range(len(metrics)):
    axes[0, 1].text(i-width/2, vals[0, i]+0.01, f"{vals[0, i]:.3f}")
    axes[0, 1].text(i+width/2, vals[1, i]+0.01, f"{vals[1, i]:.3f}")

# --- Pie chart of true class distribution
counts = [conf_matrix[0].sum(), conf_matrix[1].sum()]
axes[1, 0].pie(counts, labels=["Normal","Anomaly"], autopct="%1.1f%%", startangle=90)
axes[1, 0].set_title("Test Class Distribution")

# --- Overall performance
perf = {
    "Accuracy": test_report["accuracy"],
    "Macro Precision": test_report["macro avg"]["precision"],
    "Macro Recall": test_report["macro avg"]["recall"],
    "Macro F1": test_report["macro avg"]["f1-score"],
    "Weighted Precision": test_report["weighted avg"]["precision"],
    "Weighted Recall": test_report["weighted avg"]["recall"],
    "Weighted F1": test_report["weighted avg"]["f1-score"],
}
axes[1, 1].barh(list(perf.keys()), list(perf.values()), color="skyblue")
axes[1, 1].set_xlim(0, 1)
axes[1, 1].set_title("Overall Performance Metrics")
for i, v in enumerate(perf.values()):
    axes[1, 1].text(v+0.01, i, f"{v:.3f}", va="center")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(png_dir, "svm_test_results.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(pdf_dir, "svm_test_results.pdf"), bbox_inches="tight")
plt.close()
