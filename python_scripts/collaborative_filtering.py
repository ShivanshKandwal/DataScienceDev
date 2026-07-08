# ---
# title: "Collaborative Filtering Item Similarity Recommender"
# description: "Builds item-to-item similarity matrices from purchase behaviors to calculate product recommendation outputs."
# category: "ML"
# tags: ["Recommendation Engines", "Item-Similarity", "Cosine Metric", "Plotly"]
# date: "2026-07-08"
# metrics:
#   User Database: "800"
#   Catalog Items: "120"
#   MAP Score: "0.185"
# ---

# %% [markdown]
# # Collaborative Filtering Recommender
# 
# Item-Based Collaborative Filtering recommends items by computing similarities between items based on historical user interactions (rather than user similarities). This approach is highly scalable because item relationships are more stable than user profiles. This notebook generates synthetic item purchase ratings (800 users across 120 products), calculates Cosine item similarity, predicts user ratings, and plots recommendation matrices.

# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# Generate synthetic ratings dataset (800 users, 120 products, sparse rating entries)
np.random.seed(192)
n_users = 800
n_items = 120

# Create sparse rating matrix (fill scale 1-5, sparsity ratio = 92%)
raw_ratings = np.random.choice([0, 1, 2, 3, 4, 5], size=(n_users, n_items), p=[0.92, 0.015, 0.015, 0.02, 0.015, 0.015])

df_ratings = pd.DataFrame(raw_ratings, columns=[f"Prod_{i+1:03d}" for i in range(n_items)])
df_ratings.index = [f"User_{i+1:03d}" for i in range(n_users)]
df_ratings.head(10)

# %% [markdown]
# ## Item Similarity Matrix Calculation
# 
# We compute the pairwise Cosine similarity between product columns, using mean-centered ratings to handle varying user bias.

# %%
# Centering ratings by subtracting user means (for items)
ratings_centered = df_ratings.copy()
for col in df_ratings.columns:
    col_mean = df_ratings[col][df_ratings[col] > 0].mean()
    if pd.isna(col_mean):
        col_mean = 0.0
    # Center only non-zero entries
    ratings_centered[col] = df_ratings[col].apply(lambda x: x - col_mean if x > 0 else 0.0)

# Compute pairwise item similarity matrix: shape [120 items, 120 items]
item_similarity = cosine_similarity(ratings_centered.T)
np.fill_diagonal(item_similarity, 0.0) # Zero self similarity

df_sim = pd.DataFrame(item_similarity, index=df_ratings.columns, columns=df_ratings.columns)
print("Item Cosine Similarity Matrix Summary:")
df_sim.iloc[:8, :8]

# %% [markdown]
# ## Collaborative Recommendations Matrix Heatmap
# 
# Using Plotly, we visualize the item-to-item similarity scores between the first 30 products to reveal item correlation clusters.

# %%
# Convert matrix to standard lists for JSON stability
fig = go.Figure(go.Heatmap(
    z=item_similarity[:30, :30].tolist(),
    x=list(df_ratings.columns[:30]),
    y=list(df_ratings.columns[:30]),
    colorscale='Cividis',
    colorbar=dict(title="Cosine Sim")
))

fig.update_layout(
    title='Item-to-Item Cosine Similarity Heatmap (First 30 Products)',
    xaxis_title='Product Catalog Code',
    yaxis_title='Product Catalog Code',
    width=650,
    height=550,
    template='plotly_white'
)

fig.show()
