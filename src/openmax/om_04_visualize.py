import os
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Directories ===
results_dir = "results/om"
png_dir = f'{results_dir}/png'
pdf_dir = f'{results_dir}/pdf'

os.makedirs(png_dir, exist_ok=True)
os.makedirs(pdf_dir, exist_ok=True)

# === 1. Load summary CSV ===
summary_df = pd.read_csv(os.path.join(results_dir, "training_evaluation_summary.csv"))

# === 2. Load parameter selection results ===
param_csv = pd.read_csv(os.path.join(results_dir, "parameter_selection_summary.csv"))
with open(os.path.join(results_dir, "parameter_selection_results.pkl"), "rb") as f:
    param_pkl = pickle.load(f)

# === 3. Load training evaluation results ===
with open(os.path.join(results_dir, "training_evaluation_results.json")) as f:
    train_eval = json.load(f)

cv_results = train_eval["cv_results"]

# === Figure 1: Summary Metric Bar Chart ===
plt.figure(figsize=(10, 6))
metrics = summary_df["Metric"]
x = range(len(metrics))
plt.bar(x, summary_df["CV_Mean"], yerr=summary_df["CV_Std"], label="CV", alpha=0.7)
plt.plot(x, summary_df["Final_Model"], color="red", marker="o", linestyle="--", label="Final")
plt.xticks(x, metrics, rotation=45, ha="right")
plt.ylabel("Score")
plt.title("OpenMax Summary Metrics (CV vs Final)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(png_dir, "openmax_summary_metrics.png"), dpi=300)
plt.savefig(os.path.join(pdf_dir, "openmax_summary_metrics.pdf"))
plt.close()

# === Figure 2: Hyperparameter Search Heatmap (Mean Score) ===
pivot_df = param_csv.pivot_table(
    index="hidden_dims",
    columns=["distance_type", "tailsize"],
    values="mean_score"
)

plt.figure(figsize=(12, 6))
sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="viridis")
plt.title("OpenMax Grid Search - Mean Score")
plt.ylabel("Hidden Dims")
plt.xlabel("Distance Type & Tail Size")
plt.tight_layout()
plt.savefig(os.path.join(png_dir, "openmax_grid_search.png"), dpi=300)
plt.savefig(os.path.join(pdf_dir, "openmax_grid_search.pdf"))
plt.close()

# === Figure 3: CV Metric Distribution (per run) ===
cv_df = pd.DataFrame(cv_results)
cv_df.rename(columns={
    "overall_accuracy": "Overall Accuracy",
    "known_accuracy": "Known Class Accuracy",
    "unknown_detection_rate": "Unknown Detection Rate",
    "macro_f1": "Macro F1",
    "weighted_f1": "Weighted F1"
}, inplace=True)

plt.figure(figsize=(12, 6))
cv_df.boxplot()
plt.title("OpenMax Cross-Validation Metric Distribution")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig(os.path.join(png_dir, "openmax_cv_metrics.png"), dpi=300)
plt.savefig(os.path.join(pdf_dir, "openmax_cv_metrics.pdf"))
plt.close()

# === Figure 4: Parameter Search Runtime vs Score ===
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=param_csv,
    x="time",
    y="mean_score",
    hue="distance_type",
    style="hidden_dims",
    size="tailsize",
    palette="Set2"
)
plt.xlabel("Runtime (s)")
plt.ylabel("Mean Score")
plt.title("OpenMax Parameter Search: Runtime vs Performance")
plt.tight_layout()
plt.savefig(os.path.join(png_dir, "openmax_param_runtime_score.png"), dpi=300)
plt.savefig(os.path.join(pdf_dir, "openmax_param_runtime_score.pdf"))
plt.close()


# print("Best OpenMax Params:")
# print(json.dumps(param_pkl["best_params"], indent=4))
# print(f"Best Score: {param_pkl['best_score']:.4f}")

# with open(os.path.join(results_dir, "best_openmax_config.json"), "w") as f:
#     json.dump(param_pkl["best_params"], f, indent=4)
