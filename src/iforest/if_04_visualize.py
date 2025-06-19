import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results_dir = "results/iforest"
png_dir = f'{results_dir}/png'
pdf_dir = f'{results_dir}/pdf'

# Load cross-validation results
with open(f'{results_dir}/all_cv_results.json', 'r') as f:
    cv_results = json.load(f)
    
cv_data = []
for result in cv_results:
    row = result['params'].copy()
    row.update({k: v for k, v in result.items() if k != 'params'})
    cv_data.append(row)

cv_df = pd.DataFrame(cv_data)

# Load final test evaluation results
with open(f'{results_dir}/final_test_evaluation.json', 'r') as f:
    test_results = json.load(f)
    
# Plotting the results
plt.style.use('default')
sns.set_palette("husl")

# Create Figure 1: Hyperparameter Impact on False Positive Rate
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Isolation Forest Hyperparameter Impact on False Positive Rate', fontsize=16, fontweight='bold')

# Bootstrap effect
sns.boxplot(data=cv_df, x='bootstrap', y='false_positive_rate', ax=axes[0,0])
axes[0,0].set_title('Bootstrap Effect')
axes[0,0].set_ylabel('False Positive Rate')

# Contamination effect
sns.boxplot(data=cv_df, x='contamination', y='false_positive_rate', ax=axes[0,1])
axes[0,1].set_title('Contamination Level Effect')
axes[0,1].set_ylabel('False Positive Rate')

# Max features effect
sns.boxplot(data=cv_df, x='max_features', y='false_positive_rate', ax=axes[0,2])
axes[0,2].set_title('Max Features Effect')
axes[0,2].set_ylabel('False Positive Rate')

# Max samples effect
sns.boxplot(data=cv_df, x='max_samples', y='false_positive_rate', ax=axes[1,0])
axes[1,0].set_title('Max Samples Effect')
axes[1,0].set_ylabel('False Positive Rate')

# N estimators effect
sns.boxplot(data=cv_df, x='n_estimators', y='false_positive_rate', ax=axes[1,1])
axes[1,1].set_title('Number of Estimators Effect')
axes[1,1].set_ylabel('False Positive Rate')

# Overall distribution
sns.histplot(data=cv_df, x='false_positive_rate', bins=20, ax=axes[1,2])
axes[1,2].set_title('False Positive Rate Distribution')
axes[1,2].set_xlabel('False Positive Rate')
axes[1,2].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(f'{png_dir}/hyperparameter_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{pdf_dir}/hyperparameter_analysis.pdf', bbox_inches='tight')
print("Saved: hyperparameter_analysis.png and .pdf")
plt.close()

# Create Figure 2: Contamination Level Detailed Analysis
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Contamination Level Analysis', fontsize=16, fontweight='bold')

contamination_levels = [0.01, 0.05, 0.1]
for i, cont_level in enumerate(contamination_levels):
    subset = cv_df[cv_df['contamination'] == cont_level]
    
    # Create a parameter combination string for better visualization
    subset_copy = subset.copy()
    subset_copy['param_combo'] = (subset_copy['bootstrap'].astype(str) + '_' + 
                                  subset_copy['max_features'].astype(str) + '_' + 
                                  subset_copy['max_samples'].astype(str) + '_' + 
                                  subset_copy['n_estimators'].astype(str))
    
    # Plot false positive rate for each configuration
    sns.barplot(data=subset_copy.head(10), x='param_combo', y='false_positive_rate', ax=axes[i])
    axes[i].set_title(f'Contamination = {cont_level}')
    axes[i].set_xlabel('Configuration (Bootstrap_MaxFeatures_MaxSamples_NEstimators)')
    axes[i].set_ylabel('False Positive Rate')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f'{png_dir}/contamination_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{pdf_dir}/contamination_analysis.pdf', bbox_inches='tight')
print("Saved: contamination_analysis.png and .pdf")
plt.close()

# Create Figure 3: Parameter Interaction Heatmaps
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Parameter Interaction Analysis', fontsize=16, fontweight='bold')

# Bootstrap vs Contamination
pivot_bootstrap_cont = cv_df.pivot_table(values='false_positive_rate', 
                                         index='bootstrap', 
                                         columns='contamination', 
                                         aggfunc='mean')
sns.heatmap(pivot_bootstrap_cont, annot=True, fmt='.4f', ax=axes[0,0], cmap='YlOrRd')
axes[0,0].set_title('Bootstrap vs Contamination')

# Max Features vs Max Samples
pivot_features_samples = cv_df.pivot_table(values='false_positive_rate', 
                                          index='max_features', 
                                          columns='max_samples', 
                                          aggfunc='mean')
sns.heatmap(pivot_features_samples, annot=True, fmt='.4f', ax=axes[0,1], cmap='YlOrRd')
axes[0,1].set_title('Max Features vs Max Samples')

# Contamination vs N Estimators
pivot_cont_est = cv_df.pivot_table(values='false_positive_rate', 
                                  index='contamination', 
                                  columns='n_estimators', 
                                  aggfunc='mean')
sns.heatmap(pivot_cont_est, annot=True, fmt='.4f', ax=axes[1,0], cmap='YlOrRd')
axes[1,0].set_title('Contamination vs N Estimators')

# Max Samples vs N Estimators
pivot_samples_est = cv_df.pivot_table(values='false_positive_rate', 
                                     index='max_samples', 
                                     columns='n_estimators', 
                                     aggfunc='mean')
sns.heatmap(pivot_samples_est, annot=True, fmt='.4f', ax=axes[1,1], cmap='YlOrRd')
axes[1,1].set_title('Max Samples vs N Estimators')

plt.tight_layout()
plt.savefig(f'{png_dir}/parameter_interactions.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{pdf_dir}/parameter_interactions.pdf', bbox_inches='tight')
print("Saved: parameter_interactions.png and .pdf")
plt.close()

# Create Figure 4: Best vs Worst Configurations Comparison
fig, ax = plt.subplots(1, 1, figsize=(14, 8))

# Get top 10 best and worst configurations
best_configs = cv_df.nsmallest(10, 'false_positive_rate').copy()
worst_configs = cv_df.nlargest(10, 'false_positive_rate').copy()

best_configs['config_type'] = 'Best (Lowest FPR)'
worst_configs['config_type'] = 'Worst (Highest FPR)'

comparison_df = pd.concat([best_configs, worst_configs])

# Create configuration labels
comparison_df['config_label'] = (
    'C' + comparison_df['contamination'].astype(str) + 
    '_B' + comparison_df['bootstrap'].astype(str) + 
    '_MF' + comparison_df['max_features'].astype(str) + 
    '_MS' + comparison_df['max_samples'].astype(str) + 
    '_NE' + comparison_df['n_estimators'].astype(str)
)

sns.barplot(data=comparison_df, x='config_label', y='false_positive_rate', 
           hue='config_type', ax=ax)
ax.set_title('Best vs Worst Configuration Comparison', fontsize=14, fontweight='bold')
ax.set_xlabel('Configuration (C=Contamination, B=Bootstrap, MF=MaxFeatures, MS=MaxSamples, NE=NEstimators)')
ax.set_ylabel('False Positive Rate')
ax.tick_params(axis='x', rotation=45)
ax.legend(title='Configuration Type')

plt.tight_layout()
plt.savefig(f'{png_dir}/best_vs_worst_configs.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{pdf_dir}/best_vs_worst_configs.pdf', bbox_inches='tight')
print("Saved: best_vs_worst_configs.png and .pdf")
plt.close()

# Create Figure 5: Test Results Analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Final Model Test Results Analysis', fontsize=16, fontweight='bold')

# Extract test results
test_report = test_results['test_classification_report']
confusion_matrix = np.array(test_results['test_confusion_matrix'])

# Confusion Matrix Heatmap
sns.heatmap(confusion_matrix, annot=True, fmt='d', ax=axes[0,0], 
           xticklabels=['Predicted Normal', 'Predicted Anomaly'],
           yticklabels=['True Normal', 'True Anomaly'],
           cmap='Blues')
axes[0,0].set_title('Confusion Matrix')
axes[0,0].set_ylabel('True Label')
axes[0,0].set_xlabel('Predicted Label')

# Classification Report Metrics
classes = ['Normal (Class 1)', 'Anomaly (Class 0)']
metrics = ['precision', 'recall', 'f1-score']
values = np.array([
    [test_report['1.0']['precision'], test_report['1.0']['recall'], test_report['1.0']['f1-score']],
    [test_report['0.0']['precision'], test_report['0.0']['recall'], test_report['0.0']['f1-score']]
])

# Create a bar plot for metrics by class
x = np.arange(len(metrics))
width = 0.35

axes[0,1].bar(x - width/2, values[0], width, label='Normal (Class 1)', alpha=0.8)
axes[0,1].bar(x + width/2, values[1], width, label='Anomaly (Class 0)', alpha=0.8)
axes[0,1].set_title('Classification Metrics by Class')
axes[0,1].set_xlabel('Metrics')
axes[0,1].set_ylabel('Score')
axes[0,1].set_xticks(x)
axes[0,1].set_xticklabels(metrics)
axes[0,1].legend()
axes[0,1].set_ylim(0, 1.1)

# Add value labels on bars
for i, metric in enumerate(metrics):
    axes[0,1].text(i - width/2, values[0][i] + 0.01, f'{values[0][i]:.3f}', 
                  ha='center', va='bottom', fontsize=9)
    axes[0,1].text(i + width/2, values[1][i] + 0.01, f'{values[1][i]:.3f}', 
                  ha='center', va='bottom', fontsize=9)

# Class Distribution (from confusion matrix)
true_normal = confusion_matrix[0, 0] + confusion_matrix[0, 1]
true_anomaly = confusion_matrix[1, 0] + confusion_matrix[1, 1]
class_counts = [true_normal, true_anomaly]
class_labels = ['Normal', 'Anomaly']

axes[1,0].pie(class_counts, labels=class_labels, autopct='%1.1f%%', startangle=90)
axes[1,0].set_title('Test Set Class Distribution')

# Model Performance Summary
performance_metrics = {
    'Accuracy': test_report['accuracy'],
    'Macro Avg Precision': test_report['macro avg']['precision'],
    'Macro Avg Recall': test_report['macro avg']['recall'],
    'Macro Avg F1': test_report['macro avg']['f1-score'],
    'Weighted Avg Precision': test_report['weighted avg']['precision'],
    'Weighted Avg Recall': test_report['weighted avg']['recall'],
    'Weighted Avg F1': test_report['weighted avg']['f1-score']
}

metric_names = list(performance_metrics.keys())
metric_values = list(performance_metrics.values())

bars = axes[1,1].barh(metric_names, metric_values, color='skyblue', alpha=0.7)
axes[1,1].set_title('Overall Model Performance Metrics')
axes[1,1].set_xlabel('Score')
axes[1,1].set_xlim(0, 1)

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, metric_values)):
    axes[1,1].text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'{png_dir}/test_results_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{pdf_dir}/test_results_analysis.pdf', bbox_inches='tight')
print("Saved: test_results_analysis.png and .pdf")
plt.close()

# Create Figure 6: Research Summary Dashboard
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

fig.suptitle('Isolation Forest Research Summary Dashboard', fontsize=18, fontweight='bold')

# 1. Hyperparameter Importance (based on variance in FPR)
ax1 = fig.add_subplot(gs[0, 0])
param_importance = {}
for param in ['contamination', 'bootstrap', 'max_features', 'max_samples', 'n_estimators']:
    if cv_df[param].dtype in ['bool']:
        param_variance = cv_df.groupby(param)['false_positive_rate'].var().mean()
    else:
        param_variance = cv_df.groupby(param)['false_positive_rate'].var().mean()
    param_importance[param] = param_variance

param_names = list(param_importance.keys())
param_vars = list(param_importance.values())
bars = ax1.bar(param_names, param_vars, color='lightcoral', alpha=0.7)
ax1.set_title('Parameter Importance\n(Based on FPR Variance)')
ax1.set_ylabel('Variance in FPR')
ax1.tick_params(axis='x', rotation=45)

# 2. Best Configuration Details
ax2 = fig.add_subplot(gs[0, 1])
best_config = cv_df.loc[cv_df['false_positive_rate'].idxmin()]
best_params_viz = {
    'contamination': best_config['contamination'],
    'bootstrap': best_config['bootstrap'],
    'max_features': best_config['max_features'],
    'max_samples': best_config['max_samples'],
    'n_estimators': best_config['n_estimators']
}

ax2.axis('off')
ax2.text(0.1, 0.9, 'Best Configuration', fontsize=14, fontweight='bold', transform=ax2.transAxes)
ax2.text(0.1, 0.8, f"FPR: {best_config['false_positive_rate']:.6f}", fontsize=12, transform=ax2.transAxes)
y_pos = 0.7
for param, value in best_params_viz.items():
    ax2.text(0.1, y_pos, f"{param}: {value}", fontsize=10, transform=ax2.transAxes)
    y_pos -= 0.1

# 3. FPR Distribution by Contamination
ax3 = fig.add_subplot(gs[0, 2:])
for cont_level in [0.01, 0.05, 0.1]:
    subset = cv_df[cv_df['contamination'] == cont_level]
    ax3.hist(subset['false_positive_rate'], alpha=0.6, label=f'Contamination = {cont_level}', bins=15)
ax3.set_title('False Positive Rate Distribution by Contamination Level')
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('Frequency')
ax3.legend()

# 4. Confusion Matrix (normalized)
ax4 = fig.add_subplot(gs[1, 0])
conf_matrix_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
sns.heatmap(conf_matrix_norm, annot=True, fmt='.3f', ax=ax4, 
           xticklabels=['Pred Normal', 'Pred Anomaly'],
           yticklabels=['True Normal', 'True Anomaly'],
           cmap='Blues')
ax4.set_title('Normalized Confusion Matrix')

# 5. ROC-like Analysis (TPR vs FPR)
ax5 = fig.add_subplot(gs[1, 1])
# Calculate TPR and FPR from confusion matrix
tn, fp, fn, tp = confusion_matrix.ravel()
tpr = tp / (tp + fn)  # True Positive Rate (Recall)
fpr = fp / (fp + tn)  # False Positive Rate
specificity = tn / (tn + fp)  # True Negative Rate

metrics_roc = ['TPR\n(Sensitivity)', 'TNR\n(Specificity)', 'PPV\n(Precision)', 'NPV']
values_roc = [
    tpr, 
    specificity, 
    test_report['0.0']['precision'] if test_report['0.0']['precision'] > 0 else 0.001,  # Handle 0 precision
    tn / (tn + fn)  # Negative Predictive Value
]

bars = ax5.bar(metrics_roc, values_roc, color=['green', 'blue', 'orange', 'red'], alpha=0.7)
ax5.set_title('Classification Performance Metrics')
ax5.set_ylabel('Score')
ax5.set_ylim(0, 1)
for bar, value in zip(bars, values_roc):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{value:.3f}', 
             ha='center', va='bottom', fontsize=9)

# 6. Parameter Sensitivity Analysis
ax6 = fig.add_subplot(gs[1, 2:])
param_means = cv_df.groupby(['contamination', 'bootstrap'])['false_positive_rate'].mean().unstack()
sns.heatmap(param_means, annot=True, fmt='.4f', ax=ax6, cmap='RdYlBu_r')
ax6.set_title('Mean FPR: Contamination vs Bootstrap')
ax6.set_xlabel('Bootstrap')
ax6.set_ylabel('Contamination')

# 7. Model Complexity vs Performance
ax7 = fig.add_subplot(gs[2, 0])
complexity_score = cv_df['n_estimators'] * cv_df['max_features'] * cv_df['max_samples']
ax7.scatter(complexity_score, cv_df['false_positive_rate'], 
           c=cv_df['contamination'], cmap='viridis', alpha=0.6)
ax7.set_xlabel('Model Complexity Score')
ax7.set_ylabel('False Positive Rate')
ax7.set_title('Model Complexity vs Performance')
cbar = plt.colorbar(ax7.collections[0], ax=ax7)
cbar.set_label('Contamination Level')

# 8. Performance Summary Table
ax8 = fig.add_subplot(gs[2, 1:])
ax8.axis('off')

# Create summary statistics table
summary_stats = pd.DataFrame({
    'Metric': ['Total Experiments', 'Best FPR', 'Mean FPR', 'Std FPR', 
               'Test Accuracy', 'Anomaly Recall', 'Normal Precision', 'F1-Score (Normal)'],
    'Value': [
        len(cv_results),
        f"{cv_df['false_positive_rate'].min():.6f}",
        f"{cv_df['false_positive_rate'].mean():.6f}",
        f"{cv_df['false_positive_rate'].std():.6f}",
        f"{test_report['accuracy']:.3f}",
        f"{test_report['0.0']['recall']:.3f}",
        f"{test_report['1.0']['precision']:.3f}",
        f"{test_report['1.0']['f1-score']:.3f}"
    ]
})

# Create table
table = ax8.table(cellText=summary_stats.values,
                 colLabels=summary_stats.columns,
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.4, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)
ax8.set_title('Model Performance Summary', fontsize=12, fontweight='bold', pad=20)

plt.savefig(f'{png_dir}/research_summary_dashboard.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{pdf_dir}/research_summary_dashboard.pdf', bbox_inches='tight')
print("Saved: research_summary_dashboard.png and .pdf")
plt.close()