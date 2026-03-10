# Step 1: Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Step 2: Load Iris dataset
iris = load_iris()
X = iris.data[:, [0,2]]   # Sepal Length and Petal Length

# Step 3: Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# Step 4: Create dataframe
df = pd.DataFrame(X, columns=["Sepal Length","Petal Length"])
df["Cluster Label"] = labels

# Step 5: Select sample points similar to lab manual
sample = df.iloc[[0,1,50,51,100,101]].copy()
sample["Data Point"] = ["P1","P2","P3","P4","P5","P6"]

# Format cluster names
sample["Cluster Label"] = "Cluster " + sample["Cluster Label"].astype(str)

sample = sample[["Data Point","Sepal Length","Petal Length","Cluster Label"]]

print(sample)

# Step 6: Visualization
centroids = kmeans.cluster_centers_

plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')

plt.scatter(centroids[:,0], centroids[:,1],
            s=200, c='red', marker='X')

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("K-Means Clustering on Iris Dataset")

plt.show()