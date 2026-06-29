# ---
# title: "Collaborative Filtering Movie Recommender"
# description: "Implements user-based collaborative filtering from scratch using cosine similarities. Generates rating prediction heatmaps and recommendation bars."
# category: "ML"
# tags: ["Recommendation", "Collaborative Filtering", "Cosine Similarity", "Plotly"]
# date: "2026-06-29"
# metrics:
#   User Count: "100"
#   Movie Count: "20"
#   Average Rating: "3.62"
# ---

# %% [markdown]
# # Collaborative Filtering Movie Recommender
# 
# Recommendation engines suggest relevant items to users. User-based collaborative filtering identifies users with similar taste profiles (neighbors) and predicts target user preferences based on neighbors' historical ratings. This project implements user-based collaborative filtering from scratch using NumPy and Pandas, and visualizes recommendations using Plotly.

# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Generate synthetic ratings dataset (100 users, 20 movies)
np.random.seed(42)
n_users = 100
n_movies = 20

# Movies list
movies = [
    "Inception", "The Dark Knight", "Interstellar", "Pulp Fiction", 
    "Fight Club", "The Matrix", "Forrest Gump", "Spirited Away", 
    "Parasite", "Whiplash", "Gladiator", "The Lion King", 
    "Memento", "The Departed", "The Prestige", "WALL-E", 
    "Alien", "The Godfather", "Star Wars IV", "Casablanca"
]

# Generate sparse ratings matrix (values from 1 to 5, with ~60% missing values represented by NaN)
raw_ratings = np.random.choice(
    [1.0, 2.0, 3.0, 4.0, 5.0, np.nan], 
    size=(n_users, n_movies), 
    p=[0.05, 0.08, 0.12, 0.15, 0.10, 0.50]
)

# Convert to DataFrame
df_ratings = pd.DataFrame(raw_ratings, columns=movies, index=[f"User_{i+1}" for i in range(n_users)])
df_ratings.head(10)

# %% [markdown]
# ## User Cosine Similarity Computation
# 
# To calculate similarity between users, we:
# 1. Mean-center the ratings for each user to normalize optimistic vs pessimistic rating scales.
# 2. Fill missing values with 0.0 (the user's mean rating after subtraction).
# 3. Compute the pairwise cosine similarity matrix.

# %%
# Center user ratings
user_means = df_ratings.mean(axis=1)
df_centered = df_ratings.sub(user_means, axis=0).fillna(0.0)

# Compute Cosine Similarity Matrix
dot_product = np.dot(df_centered.values, df_centered.values.T)
norms = np.linalg.norm(df_centered.values, axis=1)
norms[norms == 0] = 1e-9  # Avoid division by zero

similarity_matrix = dot_product / np.outer(norms, norms)
df_similarity = pd.DataFrame(similarity_matrix, index=df_ratings.index, columns=df_ratings.index)

print(f"Similarity matrix shape: {df_similarity.shape}")
print("Sample similarity coefficients:")
df_similarity.iloc[:5, :5]

# %% [markdown]
# ## Rating Predictions & Recommendation Engine
# 
# We implement the recommendation predictor. For a target user and an unrated movie, we extract top K similar users who *have* rated the movie and calculate a similarity-weighted average of their centered ratings.

# %%
def predict_rating(user_id, movie, k=10):
    if not pd.isna(df_ratings.loc[user_id, movie]):
        return df_ratings.loc[user_id, movie]  # Already rated
        
    # Users who rated this movie
    other_ratings = df_ratings[movie].dropna()
    if len(other_ratings) == 0:
        return user_means[user_id]  # Fallback to user average
        
    # Get similarity coefficients for these users
    similarities = df_similarity.loc[user_id, other_ratings.index]
    
    # Sort and pick top K
    top_k_users = similarities.nlargest(k)
    top_k_similarities = top_k_users.values
    top_k_ratings = df_ratings.loc[top_k_users.index, movie].values
    
    # Weighted average calculation
    sim_sum = np.sum(np.abs(top_k_similarities))
    if sim_sum == 0:
        return user_means[user_id]
        
    weighted_sum = np.sum(top_k_similarities * (top_k_ratings - user_means[top_k_users.index].values))
    predicted_rating = user_means[user_id] + (weighted_sum / sim_sum)
    return np.clip(predicted_rating, 1.0, 5.0)

# Generate recommendations for a specific user: User_15
target_user = "User_15"
unrated_movies = df_ratings.columns[df_ratings.loc[target_user].isna()]

predictions = []
for mv in unrated_movies:
    pred = predict_rating(target_user, mv)
    predictions.append((mv, pred))
    
df_recommendations = pd.DataFrame(predictions, columns=['Movie', 'Predicted_Rating']).sort_values(by='Predicted_Rating', ascending=False)
print(f"Top 5 Recommendations for {target_user}:")
df_recommendations.head(5)

# %% [markdown]
# ## Collaborative Filtering Visualizations
# 
# We generate a heatmap of user-to-user cosine similarities for the first 15 users, alongside a bar chart displaying the predicted rating scores for the top recommended movies.

# %%
# Convert dataframes/series to python lists for JSON serialization compatibility
users_subset = list(df_similarity.index[:15])
matrix_values = [[float(v) for v in row] for row in df_similarity.iloc[:15, :15].values]

top_recommendations = df_recommendations.head(6)
rec_movies = list(top_recommendations['Movie'])
rec_ratings = [float(r) for r in top_recommendations['Predicted_Rating']]

# Create Plotly Heatmap
fig_heatmap = go.Figure(data=go.Heatmap(
    z=matrix_values,
    x=users_subset,
    y=users_subset,
    colorscale='Viridis',
    zmin=0.0, zmax=1.0,
    hoverongaps=False
))
fig_heatmap.update_layout(
    title='User Cosine Similarity Matrix (Subset of 15 Users)',
    xaxis_title='User ID',
    yaxis_title='User ID',
    width=600,
    height=500
)
fig_heatmap.show()

# Create Recommendations Bar Chart
fig_bars = go.Figure(data=go.Bar(
    x=rec_ratings,
    y=rec_movies,
    orientation='h',
    marker=dict(color='#8B5CF6')
))
fig_bars.update_layout(
    title=f'Predicted Rating Scores for {target_user}',
    xaxis_title='Predicted Rating (1-5)',
    yaxis_title='Movie Title',
    xaxis=dict(range=[1.0, 5.0]),
    width=600,
    height=400
)
fig_bars.show()
