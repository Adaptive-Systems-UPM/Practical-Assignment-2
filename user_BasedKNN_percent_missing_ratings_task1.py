import pickle
from surprise import KNNWithMeans
from surprise import Dataset
from surprise.accuracy import mae
from surprise.model_selection import train_test_split


# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')

# Dataset splitting for 25% sparsity (test_size=0.25)
trainset25, testset25 = train_test_split(data, test_size=.25, random_state=42)

# test different K values
k_values = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 250]
mae_results_25 = {}

print("=" * 60)
print("Task 1a: Finding Optimal K with 25% Missing Ratings")
print("=" * 60)

for k in k_values:
    # Configure KNN with Pearson similarity for user-based CF
    sim_options_KNN = {'name': "pearson",
                       'user_based': True  # compute similarities between users
                       }

    # prepare user-based KNN for predicting ratings from trainset25
    algo = KNNWithMeans(k, sim_options=sim_options_KNN, verbose=False)
    algo.fit(trainset25)

    # estimate the ratings for all the pairs (user, item) in testset25
    predictions25KNN = algo.test(testset25)

    # calculate MAE
    mae_value = mae(predictions25KNN, verbose=False)
    mae_results_25[k] = mae_value

    print(f"K = {k:3d} | MAE = {mae_value:.4f}")

# find optimal K
best_k_25 = min(mae_results_25, key=mae_results_25.get)
print("\n" + "=" * 60)
print(f"Best K for 25% sparsity: {best_k_25}")
print(f"Best MAE: {mae_results_25[best_k_25]:.4f}")
print("=" * 60)

# Show diminishing returns
print("\nDiminishing Returns Analysis:")
for i in range(1, len(k_values)):
    k_curr = k_values[i]
    k_prev = k_values[i-1]
    improvement = mae_results_25[k_prev] - mae_results_25[k_curr]
    print(f"K {k_prev:3d} → {k_curr:3d}: Improvement = {improvement:.4f}")

# Dataset splitting for 75% sparsity
trainset75, testset75 = train_test_split(data, test_size=.75, random_state=42)

mae_results_75 = {}

print("\n" + "=" * 60)
print("Task 1b: Finding Optimal K with 75% Missing Ratings")
print("=" * 60)


for k in k_values:
    sim_options = {'name': 'pearson', 'user_based': True}
    algo = KNNWithMeans(k=k, sim_options=sim_options, verbose=False)
    algo.fit(trainset75)
    predictions = algo.test(testset75)
    mae_value = mae(predictions, verbose=False)
    mae_results_75[k] = mae_value
    print(f"K = {k:3d} | MAE = {mae_value:.4f}")

best_k_75 = min(mae_results_75, key=mae_results_75.get)
print(f"\nBest K (75%): {best_k_75} with MAE = {mae_results_75[best_k_75]:.4f}")

# Diminishing returns analysis for 75% sparsity
print("\nDiminishing Returns Analysis for 75% sparsity:")
for i in range(1, len(k_values)):
    k_curr = k_values[i]
    k_prev = k_values[i-1]
    improvement = mae_results_75[k_prev] - mae_results_75[k_curr]
    print(f"K {k_prev:3d} → {k_curr:3d}: Improvement = {improvement:.4f}")

# Save all KNN results to a file
knn_results = {
    'best_k_25': best_k_25,
    'best_k_75': best_k_75,
    'mae_results_25': mae_results_25,
    'mae_results_75': mae_results_75,
    'trainset25': trainset25,
    'testset25': testset25,
    'trainset75': trainset75,
    'testset75': testset75
}

with open('knn_results.pkl', 'wb') as f:
    pickle.dump(knn_results, f)

print("KNN results saved to 'knn_results.pkl'")