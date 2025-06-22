import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os


results_dir = "results/ae"
png_dir = f'{results_dir}/png'
pdf_dir = f'{results_dir}/pdf'
os.makedirs(png_dir, exist_ok=True)
os.makedirs(pdf_dir, exist_ok=True)

with open(os.path.join(results_dir, "all_cv_results.json"), "r") as f:
    cv_results = json.load(f)

cv_data = []
for result in cv_results:
    row = result["params"].copy()
    # Flatten avg_false_positive_rate
    row["false_positive_rate"] = result.get("avg_false_positive_rate", np.nan)
    cv_data.append(row)
cv_df = pd.DataFrame(cv_data)

# === Load final test results ===
with open(os.path.join(results_dir, "final_test_evaluation.json"), "r") as f:
    test_results = json.load(f)

test_report = test_results["test_classification_report"]
conf_matrix = np.array(test_results["test_confusion_matrix"])

# === Plotting setup ===
plt.style.use("default")
sns.set_palette("husl")

# === Figure 1: Hyperparameter Impact on FPR ===
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Autoencoder Hyperparameter Impact on False Positive Rate", fontsize=16, fontweight="bold")

sns.boxplot(data=cv_df, x="batch_size", y="false_positive_rate", ax=axes[0, 0])
axes[0, 0].set_title("Batch Size Effect")

sns.boxplot(data=cv_df, x="contamination", y="false_positive_rate", ax=axes[0, 1])
axes[0, 1].set_title("Contamination Level Effect")

sns.boxplot(data=cv_df, x="epochs", y="false_positive_rate", ax=axes[0, 2])
axes[0, 2].set_title("Epochs Effect")

sns.boxplot(data=cv_df, x="hidden_dim", y="false_positive_rate", ax=axes[1, 0])
axes[1, 0].set_title("Hidden Dimension Effect")

sns.boxplot(data=cv_df, x="loss_fn", y="false_positive_rate", ax=axes[1, 1])
axes[1, 1].set_title("Loss Function Effect")

sns.histplot(data=cv_df, x="false_positive_rate", bins=20, ax=axes[1, 2])
axes[1, 2].set_title("False Positive Rate Distribution")

for ax in axes.flatten():
    ax.set_xlabel("")
    ax.set_ylabel("False Positive Rate")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(png_dir, "hyperparameter_impact.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(pdf_dir, "hyperparameter_impact.pdf"), bbox_inches="tight")
plt.close()

# === Figure 2: Contamination-Level Combinations ===
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Contamination-Level Analysis (Top 10 configs)", fontsize=16, fontweight="bold")
for i, cont in enumerate(sorted(cv_df["contamination"].unique())):
    subset = cv_df[cv_df["contamination"] == cont]
    subset = subset.sort_values("false_positive_rate").head(10).copy()
    subset["combo"] = (
        subset.batch_size.astype(str) + "_ep" + subset.epochs.astype(str) +
        "_hd" + subset.hidden_dim.astype(str) + "_" + subset.loss_fn
    )
    sns.barplot(data=subset, x="combo", y="false_positive_rate", ax=axes[i])
    axes[i].set_title(f"Contamination = {cont}")
    axes[i].tick_params(axis="x", rotation=45)
    axes[i].set_xlabel("Param combination")
    axes[i].set_ylabel("False Positive Rate")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(png_dir, "contamination_analysis.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(pdf_dir, "contamination_analysis.pdf"), bbox_inches="tight")
plt.close()

# === Figure 3: Parameter Interaction Heatmaps ===
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Parameter Interaction Analysis", fontsize=16, fontweight="bold")

# batch_size vs epochs
pivot_be = cv_df.pivot_table(values="false_positive_rate", index="batch_size", columns="epochs", aggfunc="mean")
sns.heatmap(pivot_be, annot=True, fmt=".4f", cmap="YlOrRd", ax=axes[0, 0])
axes[0, 0].set_title("Batch Size vs Epochs")

# batch_size vs hidden_dim
pivot_bh = cv_df.pivot_table(values="false_positive_rate", index="batch_size", columns="hidden_dim", aggfunc="mean")
sns.heatmap(pivot_bh, annot=True, fmt=".4f", cmap="YlOrRd", ax=axes[0, 1])
axes[0, 1].set_title("Batch Size vs Hidden Dim")

# epochs vs hidden_dim
pivot_eh = cv_df.pivot_table(values="false_positive_rate", index="epochs", columns="hidden_dim", aggfunc="mean")
sns.heatmap(pivot_eh, annot=True, fmt=".4f", cmap="YlOrRd", ax=axes[1, 0])
axes[1, 0].set_title("Epochs vs Hidden Dim")

# loss_fn vs batch_size
pivot_lb = cv_df.pivot_table(values="false_positive_rate", index="loss_fn", columns="batch_size", aggfunc="mean")
sns.heatmap(pivot_lb, annot=True, fmt=".4f", cmap="YlOrRd", ax=axes[1, 1])
axes[1, 1].set_title("Loss Fn vs Batch Size")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(png_dir, "parameter_interactions.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(pdf_dir, "parameter_interactions.pdf"), bbox_inches="tight")
plt.close()

# === Figure 4: Best vs Worst Configurations Comparison ===
best = cv_df.nsmallest(10, "false_positive_rate").copy()
worst = cv_df.nlargest(10, "false_positive_rate").copy()
best["type"], worst["type"] = "Best", "Worst"
comp = pd.concat([best, worst])
comp["label"] = (
    "C"+comp.contamination.astype(str)+
    "_B"+comp.batch_size.astype(str)+
    "_E"+comp.epochs.astype(str)+
    "_H"+comp.hidden_dim.astype(str)+
    "_L"+comp.loss_fn
)
plt.figure(figsize=(14, 8))
sns.barplot(data=comp, x="label", y="false_positive_rate", hue="type")
plt.xticks(rotation=45)
plt.ylabel("False Positive Rate")
plt.xlabel("Config")
plt.title("Best vs Worst Configs")
plt.legend(title="")
plt.tight_layout()
plt.savefig(os.path.join(png_dir, "best_vs_worst.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(pdf_dir, "best_vs_worst.pdf"), bbox_inches="tight")
plt.close()

# === Figure 5: Test Results Analysis ===
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Final Autoencoder Test Results", fontsize=16, fontweight="bold")

sns.heatmap(conf_matrix, annot=True, fmt="d",
            xticklabels=["Pred Normal","Pred Anomaly"],
            yticklabels=["True Normal","True Anomaly"],
            cmap="Blues", ax=axes[0, 0])
axes[0, 0].set_title("Confusion Matrix")

# Metrics bar plot
metrics = ["precision", "recall", "f1-score"]
norm = ["1.0", "0.0"]
vals = np.array([[test_report[c][m] for m in metrics] for c in norm])
x = np.arange(len(metrics))
width = 0.35
axes[0, 1].bar(x - width/2, vals[0], width, label="Normal")
axes[0, 1].bar(x + width/2, vals[1], width, label="Anomaly")
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(metrics)
axes[0, 1].legend()
axes[0, 1].set_ylim(0, 1)
axes[0, 1].set_title("Classification Metrics by Class")

for i in range(len(metrics)):
    axes[0, 1].text(i-width/2, vals[0, i]+0.01, f"{vals[0, i]:.3f}")
    axes[0, 1].text(i+width/2, vals[1, i]+0.01, f"{vals[1, i]:.3f}")

# Class distribution pie
counts = [conf_matrix[0].sum(), conf_matrix[1].sum()]
axes[1, 0].pie(counts, labels=["Normal","Anomaly"], autopct="%1.1f%%", startangle=90)
axes[1, 0].set_title("Test Class Distribution")

# Overall metrics
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
plt.savefig(os.path.join(png_dir, "test_results.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(pdf_dir, "test_results.pdf"), bbox_inches="tight")
plt.close()