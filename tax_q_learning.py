import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load clustered state data
df = pd.read_csv("data/state_clusters.csv")

# Create scatter plot of row/col colored by cluster
plt.figure(figsize=(6, 6))
scatter = plt.scatter(df["col"], df["row"], c=df["cluster"], cmap="tab10", alpha=1.0, s=100)

# Set tick intervals to whole numbers (0 to 4)
plt.xticks(ticks=np.arange(5))
plt.yticks(ticks=np.arange(5))

plt.gca().invert_yaxis()  # Top-left is (0, 0)
plt.xlabel("Column")
plt.ylabel("Row")
plt.title("Taxi-v3 Grid Clusters (based on Q-values)")
plt.colorbar(scatter, label="Cluster")
plt.grid(True)
plt.tight_layout()
plt.show()
