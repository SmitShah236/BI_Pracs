import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate some random data for clustering
X, _ = make_blobs(n_samples=150, centers=4, random_state=42)

# Create a KMeans clustering model
k = 4  # Number of clusters
model = KMeans(n_clusters=k)
model.fit(X)

# Get cluster centers and labels
cluster_centers = model.cluster_centers_
labels = model.labels_

# Plot the data points colored by cluster
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=20)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=150, label='Cluster Centers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()