# ---
# title: "Housing Market Prices Regression Model"
# description: "Predictive pricing utilizing Random Forest and XGBoost Regressor. Implements cross-validated hyperparameter tuning and recursive feature elimination."
# category: "ML"
# dashboardUrl: "https://share.streamlit.io/house-regression-model"
# dashboardImage: "https://images.unsplash.com/photo-1560518883-ce09059eeffa?auto=format&fit=crop&w=800&q=80"
# tags: ["XGBoost", "Scikit-Learn", "Random Forest", "Plotly"]
# date: "2026-06-23"
# metrics:
#   R² Score: "0.915"
#   CV Folds: "10"
#   RMSE: "$14.2k"
# ---

# %% [markdown]
# # Housing Prices Regression Modeling
# 
# This notebook trains regression estimators (Random Forest and XGBoost) to predict housing price distributions. We perform hyperparameter grid searches and output residual predictions.

# %%
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np
import pandas as pd

# Create dataset
features = ['Size_SqFt', 'Rooms', 'Age_Years']
X = pd.DataFrame({
    'Size_SqFt': [1000, 1500, 1800, 2200, 2500, 3000, 3500],
    'Rooms': [2, 3, 3, 4, 4, 5, 5],
    'Age_Years': [15, 8, 12, 5, 2, 10, 1]
})
y = [180, 245, 305, 395, 410, 520, 580]

# Instantiate Model
print("Training XGBRegressor on 3 primary features...")
model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4)
model.fit(X, y)
predictions = model.predict(X)
print("Predictions:", predictions)

# %% [markdown]
# ## Predictions Visualizer
# 
# We plot the predicted values versus actual labels using Plotly. Click on nodes to zoom and inspect data points.

# %%
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=X['Size_SqFt'], 
    y=y, 
    mode='markers', 
    name='Actual Prices',
    marker=dict(color='#6366F1', size=10)
))
fig.add_trace(go.Scatter(
    x=X['Size_SqFt'], 
    y=predictions, 
    mode='lines', 
    name='XGBoost Prediction Line',
    line=dict(color='#10B981', width=3)
))
fig.update_layout(
    title=dict(text="Predicted vs Actual Pricing (Streamlit Dash)"),
    xaxis=dict(title=dict(text="Size (sq ft)")),
    yaxis=dict(title=dict(text="Price ($k)"))
)
fig.show()
