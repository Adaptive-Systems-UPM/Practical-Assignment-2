from surprise import Dataset
from surprise.model_selection import train_test_split
import pickle
from surprise import SVD
from surprise.accuracy import mae

# ============= Load KNN Results from Task 1 =============
print("=" * 70)
print("Loading KNN results from Task 1...")
print("=" * 70)

with open('knn_results.pkl', 'rb') as f:
    knn_results = pickle.load(f)

# Extract variables
best_k_25 = knn_results['best_k_25']
best_k_75 = knn_results['best_k_75']
mae_results_25 = knn_results['mae_results_25']
mae_results_75 = knn_results['mae_results_75']
trainset25 = knn_results['trainset25']
testset25 = knn_results['testset25']
trainset75 = knn_results['trainset75']
testset75 = knn_results['testset75']

print(f"Loaded: Best K (25%) = {best_k_25}, MAE = {mae_results_25[best_k_25]:.4f}")
print(f"Loaded: Best K (75%) = {best_k_75}, MAE = {mae_results_75[best_k_75]:.4f}")

print("\n" + "=" * 70)
print("TASK 2: SVD vs KNN Comparison")
print("=" * 70)

# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')

# Dataset splitting for 25% sparsity (test_size=0.25)
trainset25, testset25 = train_test_split(data, test_size=.25, random_state=42)

# Test different numbers of latent factors
n_factors_list = [5, 10, 20, 30, 50, 75, 100, 150, 200]

# ============= SVD with 25% Sparsity =============
print("\n--- SVD with 25% Missing Ratings ---")
print("-" * 70)
svd_results_25 = {}

for n_factors in n_factors_list:
    algo_svd = SVD(n_factors=n_factors, n_epochs=20, random_state=42)
    algo_svd.fit(trainset25)
    predictions = algo_svd.test(testset25)
    mae_value = mae(predictions, verbose=False)
    svd_results_25[n_factors] = mae_value
    print(f"Factors = {n_factors:3d} | MAE = {mae_value:.4f}")

best_factors_25 = min(svd_results_25, key=svd_results_25.get)
print(f"\nBest SVD (25%): {best_factors_25} factors, MAE = {svd_results_25[best_factors_25]:.4f}")
print(f"Best KNN (25%): K = {best_k_25}, MAE = {mae_results_25[best_k_25]:.4f}")
print(f"SVD improvement over KNN: {mae_results_25[best_k_25] - svd_results_25[best_factors_25]:.4f}")

# ============= SVD with 75% Sparsity =============
print("\n--- SVD with 75% Missing Ratings ---")
print("-" * 70)
svd_results_75 = {}

for n_factors in n_factors_list:
    algo_svd = SVD(n_factors=n_factors, n_epochs=20, random_state=42)
    algo_svd.fit(trainset75)
    predictions = algo_svd.test(testset75)
    mae_value = mae(predictions, verbose=False)
    svd_results_75[n_factors] = mae_value
    print(f"Factors = {n_factors:3d} | MAE = {mae_value:.4f}")

best_factors_75 = min(svd_results_75, key=svd_results_75.get)
print(f"\nBest SVD (75%): {best_factors_75} factors, MAE = {svd_results_75[best_factors_75]:.4f}")
print(f"Best KNN (75%): K = {best_k_75}, MAE = {mae_results_75[best_k_75]:.4f}")
print(f"SVD improvement over KNN: {mae_results_75[best_k_75] - svd_results_75[best_factors_75]:.4f}")

# ============= Summary Comparison =============
print("\n" + "=" * 70)
print("SUMMARY: SVD vs KNN Comparison")
print("=" * 70)
print(f"\n25% Sparsity:")
print(f"  KNN (K={best_k_25}):          MAE = {mae_results_25[best_k_25]:.4f}")
print(f"  SVD ({best_factors_25} factors): MAE = {svd_results_25[best_factors_25]:.4f}")
print(f"  Improvement:          {mae_results_25[best_k_25] - svd_results_25[best_factors_25]:.4f} ({((mae_results_25[best_k_25] - svd_results_25[best_factors_25])/mae_results_25[best_k_25]*100):.2f}%)")

print(f"\n75% Sparsity:")
print(f"  KNN (K={best_k_75}):          MAE = {mae_results_75[best_k_75]:.4f}")
print(f"  SVD ({best_factors_75} factors): MAE = {svd_results_75[best_factors_75]:.4f}")
print(f"  Improvement:          {mae_results_75[best_k_75] - svd_results_75[best_factors_75]:.4f} ({((mae_results_75[best_k_75] - svd_results_75[best_factors_75])/mae_results_75[best_k_75]*100):.2f}%)")

print("\n" + "=" * 70)
print("Saving SVD results ...")
print("=" * 70)

svd_results = {
    'best_factors_25': best_factors_25,
    'best_factors_75': best_factors_75,
    'svd_results_25': svd_results_25,
    'svd_results_75': svd_results_75
}

with open('svd_results.pkl', 'wb') as f:
    pickle.dump(svd_results, f)

print("SVD results saved to 'svd_results.pkl'")
