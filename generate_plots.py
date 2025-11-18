"""
Generate Visualizations for Assignment 2 Report
Loads results from task outputs and creates plots
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# ============= Load Results from Task Files =============
print("Loading results from task output files...")

# Load KNN results from Task 1
with open('knn_results.pkl', 'rb') as f:
    knn_data = pickle.load(f)

mae_results_25 = knn_data['mae_results_25']
mae_results_75 = knn_data['mae_results_75']

# Load SVD results from Task 2
with open('svd_results.pkl', 'rb') as f:
    svd_data = pickle.load(f)

svd_results_25 = svd_data['svd_results_25']
svd_results_75 = svd_data['svd_results_75']

# Load Task 3 results
with open('task3_results.pkl', 'rb') as f:
    task3_data = pickle.load(f)

n_values = task3_data['n_values']
knn_25_task3 = task3_data['knn_25_results']
svd_25_task3 = task3_data['svd_25_results']
knn_75_task3 = task3_data['knn_75_results']
svd_75_task3 = task3_data['svd_75_results']

print("All results loaded successfully")

# Extract data for plotting
k_values = list(mae_results_25.keys())
mae_25 = list(mae_results_25.values())
mae_75 = list(mae_results_75.values())

factors = list(svd_results_25.keys())
svd_mae_25 = list(svd_results_25.values())
svd_mae_75 = list(svd_results_75.values())

# Extract Task 3 metrics
knn_25_precision = [x[1] for x in knn_25_task3]
knn_25_recall = [x[2] for x in knn_25_task3]
knn_25_f1 = [x[3] for x in knn_25_task3]

svd_25_precision = [x[1] for x in svd_25_task3]
svd_25_recall = [x[2] for x in svd_25_task3]
svd_25_f1 = [x[3] for x in svd_25_task3]

knn_75_precision = [x[1] for x in knn_75_task3]
knn_75_recall = [x[2] for x in knn_75_task3]
knn_75_f1 = [x[3] for x in knn_75_task3]

svd_75_precision = [x[1] for x in svd_75_task3]
svd_75_recall = [x[2] for x in svd_75_task3]
svd_75_f1 = [x[3] for x in svd_75_task3]

# Get best K values
best_k_25 = min(mae_results_25, key=mae_results_25.get)
best_k_75 = min(mae_results_75, key=mae_results_75.get)
knn_best_25 = mae_results_25[best_k_25]
knn_best_75 = mae_results_75[best_k_75]

# Get best SVD factors
best_factors_25 = min(svd_results_25, key=svd_results_25.get)
best_factors_75 = min(svd_results_75, key=svd_results_75.get)

# ============= Generate All Plots =============

# Plot 1: KNN MAE vs K
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(k_values, mae_25, 'o-', linewidth=2, markersize=8, label='25% Missing (Test)', color='#2E86AB')
ax.plot(k_values, mae_75, 's-', linewidth=2, markersize=8, label='75% Missing (Test)', color='#A23B72')
ax.axvline(x=best_k_25, color='#2E86AB', linestyle='--', alpha=0.5, label=f'Best K (25%): K={best_k_25}')
ax.axvline(x=best_k_75, color='#A23B72', linestyle='--', alpha=0.5, label=f'Best K (75%): K={best_k_75}')
ax.set_xlabel('K (Number of Neighbors)', fontsize=13, fontweight='bold')
ax.set_ylabel('MAE (Mean Absolute Error)', fontsize=13, fontweight='bold')
ax.set_title('Task 1: KNN Performance - Impact of K and Data Sparsity', fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot1_knn_mae_vs_k.png', dpi=300, bbox_inches='tight')
print("Saved: plot1_knn_mae_vs_k.png")
plt.close()

# Plot 2: Diminishing Returns
improvements_25 = [mae_25[i] - mae_25[i+1] for i in range(len(mae_25)-1)]
k_transitions = [f"{k_values[i]}→{k_values[i+1]}" for i in range(len(k_values)-1)]

fig, ax = plt.subplots(figsize=(14, 7))
colors = ['#06A77D' if imp > 0 else '#D62828' for imp in improvements_25]
bars = ax.bar(range(len(improvements_25)), improvements_25, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('K Transition', fontsize=13, fontweight='bold')
ax.set_ylabel('MAE Improvement', fontsize=13, fontweight='bold')
ax.set_title('Task 1: Diminishing Returns in KNN (25% Missing)', fontsize=15, fontweight='bold')
ax.set_xticks(range(len(k_transitions)))
ax.set_xticklabels(k_transitions, rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('plot2_knn_diminishing_returns.png', dpi=300, bbox_inches='tight')
print("Saved: plot2_knn_diminishing_returns.png")
plt.close()

# Plot 3: SVD vs KNN Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 25% Sparsity
ax1.plot(factors, svd_mae_25, 'D-', linewidth=2.5, markersize=10, label='SVD', color='#F77F00')
ax1.axhline(y=knn_best_25, color='#2E86AB', linestyle='--', linewidth=2, label=f'KNN (K={best_k_25}): {knn_best_25:.4f}')
ax1.axvline(x=best_factors_25, color='#F77F00', linestyle=':', alpha=0.5, label=f'Best SVD: {best_factors_25} factors')
ax1.set_xlabel('Number of Latent Factors', fontsize=12, fontweight='bold')
ax1.set_ylabel('MAE', fontsize=12, fontweight='bold')
ax1.set_title('25% Missing Ratings', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 75% Sparsity
ax2.plot(factors, svd_mae_75, 'D-', linewidth=2.5, markersize=10, label='SVD', color='#F77F00')
ax2.axhline(y=knn_best_75, color='#A23B72', linestyle='--', linewidth=2, label=f'KNN (K={best_k_75}): {knn_best_75:.4f}')
ax2.axvline(x=best_factors_75, color='#F77F00', linestyle=':', alpha=0.5, label=f'Best SVD: {best_factors_75} factors')
ax2.set_xlabel('Number of Latent Factors', fontsize=12, fontweight='bold')
ax2.set_ylabel('MAE', fontsize=12, fontweight='bold')
ax2.set_title('75% Missing Ratings', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

fig.suptitle('Task 2: SVD vs KNN - Sparsity Mitigation', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plot3_svd_vs_knn.png', dpi=300, bbox_inches='tight')
print("Saved: plot3_svd_vs_knn.png")
plt.close()

# Plot 4: SVD Improvement Bar Chart
fig, ax = plt.subplots(figsize=(10, 7))
categories = ['25% Missing', '75% Missing']
knn_maes = [knn_best_25, knn_best_75]
svd_maes = [svd_results_25[best_factors_25], svd_results_75[best_factors_75]]
improvements = [knn_best_25 - svd_maes[0], knn_best_75 - svd_maes[1]]

x = np.arange(len(categories))
width = 0.25

bars1 = ax.bar(x - width, knn_maes, width, label='KNN', color='#2E86AB', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x, svd_maes, width, label='SVD', color='#F77F00', alpha=0.8, edgecolor='black')
bars3 = ax.bar(x + width, improvements, width, label='Improvement', color='#06A77D', alpha=0.8, edgecolor='black')

ax.set_xlabel('Sparsity Level', fontsize=13, fontweight='bold')
ax.set_ylabel('MAE', fontsize=13, fontweight='bold')
ax.set_title('Task 2: SVD Performance Advantage Over KNN', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('plot4_svd_improvement.png', dpi=300, bbox_inches='tight')
print("Saved: plot4_svd_improvement.png")
plt.close()

# Plot 5: Precision-Recall Tradeoff (25%)
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(n_values, knn_25_precision, 'o-', linewidth=2, markersize=8, label='KNN - Precision', color='#2E86AB')
ax.plot(n_values, knn_25_recall, 's-', linewidth=2, markersize=8, label='KNN - Recall', color='#06A77D')
ax.plot(n_values, knn_25_f1, 'D-', linewidth=2, markersize=8, label='KNN - F1', color='#F77F00')
ax.set_xlabel('N (Top-N Recommendations)', fontsize=13, fontweight='bold')
ax.set_ylabel('Score', fontsize=13, fontweight='bold')
ax.set_title('Task 3: Precision-Recall Tradeoff (25% Missing - KNN)', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot5_precision_recall_25.png', dpi=300, bbox_inches='tight')
print("Saved: plot5_precision_recall_25.png")
plt.close()

# Plot 6: F1 Comparison All Scenarios
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(n_values, knn_25_f1, 'o-', linewidth=2.5, markersize=9, label='KNN 25% Missing', color='#2E86AB')
ax.plot(n_values, svd_25_f1, 's-', linewidth=2.5, markersize=9, label='SVD 25% Missing', color='#06A77D')
ax.plot(n_values, knn_75_f1, '^-', linewidth=2.5, markersize=9, label='KNN 75% Missing', color='#A23B72')
ax.plot(n_values, svd_75_f1, 'D-', linewidth=2.5, markersize=9, label='SVD 75% Missing', color='#F77F00')
ax.set_xlabel('N (Top-N Recommendations)', fontsize=13, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
ax.set_title('Task 3: F1 Score Comparison Across All Scenarios', fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot6_f1_comparison_all.png', dpi=300, bbox_inches='tight')
print("Saved: plot6_f1_comparison_all.png")
plt.close()

# Plot 7: Best F1 Scores Summary
best_f1_knn_25 = max(knn_25_f1)
best_f1_svd_25 = max(svd_25_f1)
best_f1_knn_75 = max(knn_75_f1)
best_f1_svd_75 = max(svd_75_f1)

best_n_knn_25 = n_values[knn_25_f1.index(best_f1_knn_25)]
best_n_svd_25 = n_values[svd_25_f1.index(best_f1_svd_25)]
best_n_knn_75 = n_values[knn_75_f1.index(best_f1_knn_75)]
best_n_svd_75 = n_values[svd_75_f1.index(best_f1_svd_75)]

fig, ax = plt.subplots(figsize=(10, 7))
scenarios = ['KNN\n25%', 'SVD\n25%', 'KNN\n75%', 'SVD\n75%']
f1_scores = [best_f1_knn_25, best_f1_svd_25, best_f1_knn_75, best_f1_svd_75]
best_ns = [best_n_knn_25, best_n_svd_25, best_n_knn_75, best_n_svd_75]
colors_scenario = ['#2E86AB', '#06A77D', '#A23B72', '#F77F00']

bars = ax.bar(scenarios, f1_scores, color=colors_scenario, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('Best F1 Score', fontsize=13, fontweight='bold')
ax.set_title('Task 3: Best F1 Scores Summary', fontsize=15, fontweight='bold')
ax.set_ylim([0, 0.7])
ax.grid(True, alpha=0.3, axis='y')

for i, (bar, n) in enumerate(zip(bars, best_ns)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'F1={height:.4f}\nN={n}', ha='center', va='bottom',
            fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('plot7_best_f1_summary.png', dpi=300, bbox_inches='tight')
print("Saved: plot7_best_f1_summary.png")
plt.close()

print("All plots generated successfully!")
print("\nGenerated files:")
print("  1. plot1_knn_mae_vs_k.png")
print("  2. plot2_knn_diminishing_returns.png")
print("  3. plot3_svd_vs_knn.png")
print("  4. plot4_svd_improvement.png")
print("  5. plot5_precision_recall_25.png")
print("  6. plot6_f1_comparison_all.png")
print("  7. plot7_best_f1_summary.png")
