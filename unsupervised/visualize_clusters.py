import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

# Load clustered state data
df = pd.read_csv("data/state_clusters.csv")

# Define number of clusters
num_clusters = df["cluster"].nunique()

# Create discrete colormap and normalization
cmap = ListedColormap(plt.cm.tab10.colors[:num_clusters])
bounds = np.arange(num_clusters + 1) - 0.5
norm = BoundaryNorm(bounds, cmap.N)

# Plot
plt.figure(figsize=(6, 6))
scatter = plt.scatter(df["col"], df["row"], c=df["cluster"], cmap=cmap, norm=norm, s=100)

plt.xticks(ticks=np.arange(5))
plt.yticks(ticks=np.arange(5))
plt.gca().invert_yaxis()  # Top-left is (0, 0)

# Colorbar with integer ticks
cbar = plt.colorbar(scatter, ticks=np.arange(num_clusters))
cbar.set_label("Cluster")

plt.xlabel("Column")
plt.ylabel("Row")
plt.title("Taxi-v3 Grid Clusters (Discrete Colors)")
plt.grid(True)
plt.tight_layout()
plt.show()
