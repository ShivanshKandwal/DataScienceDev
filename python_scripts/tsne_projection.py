# ---
# title: "t-SNE Dimensionality Reduction Projections"
# description: "Projects high-dimensional variables down to a 2D space using t-SNE in Scikit-Learn."
# category: "ML"
# tags: ["Dimensionality Reduction", "t-SNE", "Manifold Learning", "Matplotlib"]
# date: "2026-07-08"
# metrics:
#   Input Features: "10"
#   Reduced Features: "2"
#   Perplexity Parameter: "30"
# ---

# %% [markdown]
# # t-SNE Dimensionality Reduction Projections
# 
# t-Distributed Stochastic Neighbor Embedding (t-SNE) is a non-linear dimensionality reduction technique. It calculates probability distributions of coordinate pairs in high-dimensional space and maps them to low-dimensional distributions, preserving local structural clusters. This notebook generates synthetic 10-dimensional customer coordinates (3 distinct profiles), applies t-SNE, and visualizes the 2D clusters in Matplotlib.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Generate synthetic high-dimensional data (300 samples, 10 features, 3 clusters)
np.random.seed(170)
n_samples = 300
n_features = 10

# Cluster centers in 10D space
centers = [
    np.random.uniform(-4, 4, n_features),
    np.random.uniform(-4, 4, n_features),
    np.random.uniform(-4, 4, n_features)
]

X_list, y_list = [], []
for label, center in enumerate(centers):
    noise = np.random.normal(0, 0.45, (100, n_features))
    X_list.append(center + noise)
    y_list.append(np.full(100, label))

X = np.vstack(X_list)
y = np.concatenate(y_list)

df_tsne = pd.DataFrame(X, columns=[f"Attr_{i+1:02d}" for i in range(n_features)])
df_tsne['Cluster_Label'] = y
df_tsne.head(10)

# %% [markdown]
# ## t-SNE Projection Pipeline
# 
# We fit a t-SNE manifold learning estimator to project our 10D features down to 2 principal visualization components, using a perplexity of 30.

# %%
# Train t-SNE
tsne = TSNE(n_components=2, perplexity=30.0, random_state=42)
X_embedded = tsne.fit_transform(X)

print(f"Original Feature Dimension: {X.shape}")
print(f"t-SNE Reduced Dimension:    {X_embedded.shape}")

# %% [markdown]
# ## 2D Manifold Cluster Scatter Plot
# 
# Using Matplotlib, we scatter the 2D projected coordinates, coloring markers according to their original cluster category profiles.

# %%
plt.figure(figsize=(9.5, 6.5))

colors = ['#EF4444', '#3B82F6', '#10B981']
labels = ['VIP Premium', 'Regular Shoppers', 'Bargain Seekers']

for i in range(3):
    indices = y == i
    plt.scatter(
        X_embedded[indices, 0], 
        X_embedded[indices, 1], 
        c=colors[i], 
        label=labels[i],
        edgecolors='k', 
        s=35, 
        alpha=0.8
    )

plt.title('t-SNE Non-Linear Manifold Projection: 10D customer Metrics to 2D Clusters', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('t-SNE Dimension 1', fontsize=11)
plt.ylabel('t-SNE Dimension 2', fontsize=11)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.show()
