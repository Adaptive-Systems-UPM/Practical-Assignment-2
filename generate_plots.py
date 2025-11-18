import matplotlib.pyplot as plt
import numpy as np

# set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# TASK 1: KNN Results

# Task 1a: 25% Sparsity
k_values = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 250]
mae_25 = [0.8049, 0.7717, 0.7591, 0.7533, 0.7508, 0.7488, 0.7467, 0.7458,
          0.7455, 0.7453, 0.7452, 0.7453, 0.7454, 0.7458, 0.7460, 0.7462, 0.7462]

# Task 1b: 75% Sparsity
mae_75 = [0.8421, 0.8232, 0.8186, 0.8172, 0.8167, 0.8165, 0.8164, 0.8164,
          0.8164, 0.8164, 0.8164, 0.8164, 0.8164, 0.8164, 0.8164, 0.8164, 0.8164]

# plot 1: KNN MAE vs K for both sparsity levels
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(k_values, mae_25, 'o-', linewidth=2, markersize=8, label='25% Missing (Test)', color='#2E86AB')
ax.plot(k_values, mae_75, 's-', linewidth=2, markersize=8, label='75% Missing (Test)', color='#A23B72')
ax.axvline(x=80, color='#2E86AB', linestyle='--', alpha=0.5, label='Best K (25%): K=80')
ax.axvline(x=70, color='#A23B72', linestyle='--', alpha=0.5, label='Best K (75%): K=70')
ax.set_xlabel('K (Number of Neighbors)', fontsize=13, fontweight='bold')
ax.set_ylabel('MAE (Mean Absolute Error)', fontsize=13, fontweight='bold')
ax.set_title('Task 1: KNN Performance - Impact of K and Data Sparsity', fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot1_knn_mae_vs_k.png', dpi=300, bbox_inches='tight')
print("Saved: plot1_knn_mae_vs_k.png")
plt.close()

# plot 2: diminishing Returns Analysis (25% only)
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

# TASK 2: SVD vs KNN

# SVD Results
factors = [5, 10, 20, 30, 50, 75, 100, 150, 200]
svd_mae_25 = [0.7422, 0.7403, 0.7403, 0.7404, 0.7416, 0.7410, 0.7408, 0.7431, 0.7457]
svd_mae_75 = [0.7679, 0.7683, 0.7691, 0.7700, 0.7721, 0.7736, 0.7752, 0.7801, 0.7822]

# KNN best results for comparison
knn_best_25 = 0.7452
knn_best_75 = 0.8164

# Plot 3: SVD vs KNN Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 25% Sparsity
ax1.plot(factors, svd_mae_25, 'D-', linewidth=2.5, markersize=10, label='SVD', color='#F77F00')
ax1.axhline(y=knn_best_25, color='#2E86AB', linestyle='--', linewidth=2, label=f'KNN (K=80): {knn_best_25:.4f}')
ax1.axvline(x=10, color='#F77F00', linestyle=':', alpha=0.5, label='Best SVD: 10 factors')
ax1.set_xlabel('Number of Latent Factors', fontsize=12, fontweight='bold')
ax1.set_ylabel('MAE', fontsize=12, fontweight='bold')
ax1.set_title('25% Missing Ratings', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 75% Sparsity
ax2.plot(factors, svd_mae_75, 'D-', linewidth=2.5, markersize=10, label='SVD', color='#F77F00')
ax2.axhline(y=knn_best_75, color='#A23B72', linestyle='--', linewidth=2, label=f'KNN (K=70): {knn_best_75:.4f}')
ax2.axvline(x=5, color='#F77F00', linestyle=':', alpha=0.5, label='Best SVD: 5 factors')
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

# plot 4: Improvement Bar Chart
fig, ax = plt.subplots(figsize=(10, 7))
categories = ['25% Missing', '75% Missing']
knn_maes = [knn_best_25, knn_best_75]
svd_maes = [0.7403, 0.7679]
improvements = [knn_best_25 - 0.7403, knn_best_75 - 0.7679]

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

# add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('plot4_svd_improvement.png', dpi=300, bbox_inches='tight')
print("Saved: plot4_svd_improvement.png")
plt.close()

# TASK 3: Top-N Recommendations

n_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# 25% Sparsity - KNN
knn_25_precision = [0.6299, 0.4782, 0.3860, 0.3209, 0.2728, 0.2350, 0.2049, 0.1813, 0.1623, 0.1466]
knn_25_recall = [0.6681, 0.8333, 0.9064, 0.9433, 0.9643, 0.9751, 0.9801, 0.9833, 0.9850, 0.9860]
knn_25_f1 = [0.6484, 0.6077, 0.5415, 0.4789, 0.4253, 0.3787, 0.3389, 0.3062, 0.2786, 0.2553]

# 25% Sparsity - SVD
svd_25_precision = [0.6258, 0.4761, 0.3855, 0.3209, 0.2726, 0.2344, 0.2049, 0.1812, 0.1622, 0.1466]
svd_25_recall = [0.6663, 0.8316, 0.9056, 0.9434, 0.9640, 0.9741, 0.9798, 0.9826, 0.9849, 0.9858]
svd_25_f1 = [0.6454, 0.6055, 0.5408, 0.4789, 0.4251, 0.3779, 0.3389, 0.3059, 0.2786, 0.2552]

# 75% Sparsity - KNN
knn_75_precision = [0.7053, 0.6636, 0.6066, 0.5517, 0.5043, 0.4643, 0.4314, 0.4024, 0.3774, 0.3546]
knn_75_recall = [0.3016, 0.5299, 0.6579, 0.7366, 0.7907, 0.8307, 0.8627, 0.8881, 0.9088, 0.9252]
knn_75_f1 = [0.4225, 0.5893, 0.6313, 0.6308, 0.6158, 0.5957, 0.5752, 0.5539, 0.5334, 0.5127]

# 75% Sparsity - SVD
svd_75_precision = [0.7577, 0.6929, 0.6239, 0.5647, 0.5150, 0.4734, 0.4381, 0.4084, 0.3819, 0.3586]
svd_75_recall = [0.3279, 0.5501, 0.6708, 0.7478, 0.7995, 0.8384, 0.8686, 0.8937, 0.9132, 0.9294]
svd_75_f1 = [0.4578, 0.6133, 0.6465, 0.6434, 0.6264, 0.6051, 0.5824, 0.5606, 0.5385, 0.5175]

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
fig, ax = plt.subplots(figsize=(10, 7))
scenarios = ['KNN\n25%', 'SVD\n25%', 'KNN\n75%', 'SVD\n75%']
f1_scores = [0.6484, 0.6454, 0.6313, 0.6465]
best_n = [10, 10, 30, 30]
colors_scenario = ['#2E86AB', '#06A77D', '#A23B72', '#F77F00']

bars = ax.bar(scenarios, f1_scores, color=colors_scenario, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('Best F1 Score', fontsize=13, fontweight='bold')
ax.set_title('Task 3: Best F1 Scores Summary', fontsize=15, fontweight='bold')
ax.set_ylim([0, 0.7])
ax.grid(True, alpha=0.3, axis='y')

# add value labels
for i, (bar, n) in enumerate(zip(bars, best_n)):
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
print("  1. plot1_knn_mae_vs_k.png - KNN performance vs K")
print("  2. plot2_knn_diminishing_returns.png - Diminishing returns analysis")
print("  3. plot3_svd_vs_knn.png - SVD vs KNN comparison")
print("  4. plot4_svd_improvement.png - SVD improvement bar chart")
print("  5. plot5_precision_recall_25.png - Precision-recall tradeoff")
print("  6. plot6_f1_comparison_all.png - F1 scores all scenarios")
print("  7. plot7_best_f1_summary.png - Best F1 summary")
