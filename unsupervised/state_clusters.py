import gymnasium as gym
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Load trained Q-table
q_table = np.load("data/q_table_exp1.npy")  # Ensure correct path

# Create Taxi environment
env = gym.make("Taxi-v3")

# Decode all states and collect features
state_data = []
for state in range(500):  # Total possible states
    row, col, passenger_loc, destination = env.unwrapped.decode(state)
    q_values = q_table[state]  # Shape (6,)
    state_data.append([state, row, col, passenger_loc, destination] + list(q_values))

# Create DataFrame
columns = ["state", "row", "col", "passenger", "destination"] + [f"q{i}" for i in range(6)]
df = pd.DataFrame(state_data, columns=columns)

# Apply KMeans clustering on Q-values
kmeans = KMeans(n_clusters=5, random_state=42)
df["cluster"] = kmeans.fit_predict(df[[f"q{i}" for i in range(6)]])

# Save to CSV
df.to_csv("data/state_clusters.csv", index=False)
print("✅ State cluster data saved to 'data/state_clusters.csv'")
