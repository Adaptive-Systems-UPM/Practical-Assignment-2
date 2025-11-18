import pickle
from collections import defaultdict
from surprise import KNNWithMeans, SVD
from surprise.accuracy import mae

# ============= Load Results from Previous Tasks =============
print("=" * 70)
print("Loading results from Task 1 and Task 2...")
print("=" * 70)

# Load KNN results
with open('knn_results.pkl', 'rb') as f:
    knn_results = pickle.load(f)

best_k_25 = knn_results['best_k_25']
best_k_75 = knn_results['best_k_75']
trainset25 = knn_results['trainset25']
testset25 = knn_results['testset25']
trainset75 = knn_results['trainset75']
testset75 = knn_results['testset75']

# Load SVD results
with open('svd_results.pkl', 'rb') as f:
    svd_results = pickle.load(f)

best_factors_25 = svd_results['best_factors_25']
best_factors_75 = svd_results['best_factors_75']

print(f"Loaded: Best K (25%) = {best_k_25}")
print(f"Loaded: Best K (75%) = {best_k_75}")
print(f"Loaded: Best SVD factors (25%) = {best_factors_25}")
print(f"Loaded: Best SVD factors (75%) = {best_factors_75}")


# ============= Define Precision/Recall Function =============
def precision_recall_at_n(predictions, n=10, threshold=4):
    """
    Return precision and recall at n metrics for each user.

    Relevant items are those with true rating >= threshold (4 or 5 stars).
    Recommended items are the top-N items by predicted rating.
    """
    # Map predictions to each user
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()

    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value (descending)
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items (rated >= threshold)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items that are relevant (top-N with rating >= threshold)
        n_rec_and_rel = sum(
            (true_r >= threshold)
            for (_, true_r) in user_ratings[:n]
        )

        # Precision@n: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.
        precisions[uid] = n_rec_and_rel / n if n > 0 else 0

        # Recall@n: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.
        recalls[uid] = n_rec_and_rel / n_rel if n_rel != 0 else 0

    return precisions, recalls


# ============= TASK 3: Top-N Recommendations =============
print("\n" + "=" * 70)
print("TASK 3: Top-N Recommendations (Precision, Recall, F1)")
print("=" * 70)

# N values to test
n_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Threshold for relevant items (4 or 5 stars)
threshold = 4

# ============= 25% Sparsity - KNN =============
print("\n" + "=" * 70)
print("25% Missing Ratings - KNN (K={})".format(best_k_25))
print("=" * 70)

# Train KNN model with best K
sim_options = {'name': 'pearson', 'user_based': True}
algo_knn_25 = KNNWithMeans(k=best_k_25, sim_options=sim_options, verbose=False)
algo_knn_25.fit(trainset25)
predictions_knn_25 = algo_knn_25.test(testset25)

print(f"\n{'N':>5} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
print("-" * 50)

knn_25_results = []
for n in n_values:
    precisions, recalls = precision_recall_at_n(predictions_knn_25, n=n, threshold=threshold)

    # Average over all users
    avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
    avg_recall = sum(rec for rec in recalls.values()) / len(recalls)

    # Calculate F1 score
    if avg_precision + avg_recall > 0:
        f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    else:
        f1 = 0

    knn_25_results.append((n, avg_precision, avg_recall, f1))
    print(f"{n:>5} | {avg_precision:>10.4f} | {avg_recall:>10.4f} | {f1:>10.4f}")

# ============= 25% Sparsity - SVD =============
print("\n" + "=" * 70)
print("25% Missing Ratings - SVD ({} factors)".format(best_factors_25))
print("=" * 70)

# Train SVD model with best factors
algo_svd_25 = SVD(n_factors=best_factors_25, n_epochs=20, random_state=42)
algo_svd_25.fit(trainset25)
predictions_svd_25 = algo_svd_25.test(testset25)

print(f"\n{'N':>5} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
print("-" * 50)

svd_25_results = []
for n in n_values:
    precisions, recalls = precision_recall_at_n(predictions_svd_25, n=n, threshold=threshold)

    avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
    avg_recall = sum(rec for rec in recalls.values()) / len(recalls)

    if avg_precision + avg_recall > 0:
        f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    else:
        f1 = 0

    svd_25_results.append((n, avg_precision, avg_recall, f1))
    print(f"{n:>5} | {avg_precision:>10.4f} | {avg_recall:>10.4f} | {f1:>10.4f}")

# ============= 75% Sparsity - KNN =============
print("\n" + "=" * 70)
print("75% Missing Ratings - KNN (K={})".format(best_k_75))
print("=" * 70)

# Train KNN model with best K
algo_knn_75 = KNNWithMeans(k=best_k_75, sim_options=sim_options, verbose=False)
algo_knn_75.fit(trainset75)
predictions_knn_75 = algo_knn_75.test(testset75)

print(f"\n{'N':>5} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
print("-" * 50)

knn_75_results = []
for n in n_values:
    precisions, recalls = precision_recall_at_n(predictions_knn_75, n=n, threshold=threshold)

    avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
    avg_recall = sum(rec for rec in recalls.values()) / len(recalls)

    if avg_precision + avg_recall > 0:
        f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    else:
        f1 = 0

    knn_75_results.append((n, avg_precision, avg_recall, f1))
    print(f"{n:>5} | {avg_precision:>10.4f} | {avg_recall:>10.4f} | {f1:>10.4f}")

# ============= 75% Sparsity - SVD =============
print("\n" + "=" * 70)
print("75% Missing Ratings - SVD ({} factors)".format(best_factors_75))
print("=" * 70)

# Train SVD model with best factors
algo_svd_75 = SVD(n_factors=best_factors_75, n_epochs=20, random_state=42)
algo_svd_75.fit(trainset75)
predictions_svd_75 = algo_svd_75.test(testset75)

print(f"\n{'N':>5} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
print("-" * 50)

svd_75_results = []
for n in n_values:
    precisions, recalls = precision_recall_at_n(predictions_svd_75, n=n, threshold=threshold)

    avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
    avg_recall = sum(rec for rec in recalls.values()) / len(recalls)

    if avg_precision + avg_recall > 0:
        f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    else:
        f1 = 0

    svd_75_results.append((n, avg_precision, avg_recall, f1))
    print(f"{n:>5} | {avg_precision:>10.4f} | {avg_recall:>10.4f} | {f1:>10.4f}")

# ============= Summary Analysis =============
print("\n" + "=" * 70)
print("SUMMARY: Best F1 Scores")
print("=" * 70)

# Find best F1 for each scenario
best_knn_25 = max(knn_25_results, key=lambda x: x[3])
best_svd_25 = max(svd_25_results, key=lambda x: x[3])
best_knn_75 = max(knn_75_results, key=lambda x: x[3])
best_svd_75 = max(svd_75_results, key=lambda x: x[3])

print("\n25% Sparsity:")
print(f"  KNN: Best F1 = {best_knn_25[3]:.4f} at N={best_knn_25[0]} (P={best_knn_25[1]:.4f}, R={best_knn_25[2]:.4f})")
print(f"  SVD: Best F1 = {best_svd_25[3]:.4f} at N={best_svd_25[0]} (P={best_svd_25[1]:.4f}, R={best_svd_25[2]:.4f})")

print("\n75% Sparsity:")
print(f"  KNN: Best F1 = {best_knn_75[3]:.4f} at N={best_knn_75[0]} (P={best_knn_75[1]:.4f}, R={best_knn_75[2]:.4f})")
print(f"  SVD: Best F1 = {best_svd_75[3]:.4f} at N={best_svd_75[0]} (P={best_svd_75[1]:.4f}, R={best_svd_75[2]:.4f})")
