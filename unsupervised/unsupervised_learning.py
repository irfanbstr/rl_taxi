import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load your trained Q-table
q_table = np.load("data/q_table_exp1.npy")

# Extract features for each state: best action and max Q-value
best_actions = np.argmax(q_table, axis=1)
max_q_values = np.max(q_table, axis=1)

# Create feature matrix: 2 features per state
features = np.vstack((best_actions, max_q_values)).T

# Choose number of clusters (e.g., 4)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(features)

print(f"Cluster assignments for states:\n{clusters}")

# Optional: Visualize clusters by action vs Q-value
plt.scatter(best_actions, max_q_values, c=clusters, cmap='viridis')
plt.xlabel('Best Action')
plt.ylabel('Max Q-Value')
plt.title('Clustering States by Policy Features')
plt.colorbar(label='Cluster')
plt.show()
