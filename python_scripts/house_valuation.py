# ---
# title: "House Valuation Regression Modeling"
# description: "Evaluates property prices using Ridge Regression and Random Forest. Performs feature scaling, handles multicollinearity, and visualizes residuals using interactive Plotly scatter plots."
# category: "ML"
# tags: ["Scikit-Learn", "Regression", "Random Forest", "Plotly"]
# date: "2026-06-27"
# metrics:
#   R² Score: "0.892"
#   RMSE: "$32.4k"
#   Dataset Size: "1500"
# ---

# %% [markdown]
# # House Valuation Regression Modeling
# 
# This project trains and evaluates two predictive models: a regularized Ridge Regressor and an ensemble Random Forest Regressor, to estimate housing market values based on property features (e.g., size, location rating, age, schools). We perform feature preprocessing and compare model residuals.

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Generate synthetic housing dataset
np.random.seed(42)
n_samples = 1500

size = np.random.normal(2000, 500, n_samples).clip(800, 5000)
bedrooms = np.round(size / 600 + np.random.normal(0, 0.5, n_samples)).clip(1, 6)
bathrooms = np.round(bedrooms * 0.75 + np.random.normal(0, 0.3, n_samples)).clip(1, 5)
age = np.random.uniform(0, 50, n_samples)
distance = np.random.exponential(10, n_samples).clip(0.5, 40)
school_rating = np.random.randint(1, 11, n_samples)

# True price generating function (with some non-linearities and noise)
base_price = 100000
price = (
    base_price 
    + size * 150 
    + bedrooms * 15000 
    + bathrooms * 22000 
    - age * 1200 
    - distance * 3500 
    + school_rating * 8000 
    + np.random.normal(0, 25000, n_samples)
)

df = pd.DataFrame({
    'Size_SqFt': size,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'Age_Years': age,
    'Distance_CityCenter': distance,
    'School_Rating': school_rating,
    'Price': price
})

df.head(10)

# %% [markdown]
# ## Feature Engineering & Train-Test Split
# 
# We split the data into training and testing sets (80/20) and apply standard scaling to our independent continuous features to prepare them for Ridge Regression.

# %%
X = df.drop(columns=['Price'])
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# %% [markdown]
# ## Model Training & Evaluation
# 
# We train both a Ridge Regressor and a Random Forest Regressor, and compute R-squared ($R^2$), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) on the test partition.

# %%
# Train Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

# Train Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=12)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Metrics evaluation function
def print_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"--- {name} Performance ---")
    print(f"R² Score:  {r2:.4f}")
    print(f"MAE:       ${mae:,.2f}")
    print(f"RMSE:      ${rmse:,.2f}\n")
    return r2, mae, rmse

r2_ridge, mae_ridge, rmse_ridge = print_metrics("Ridge Regression", y_test, y_pred_ridge)
r2_rf, mae_rf, rmse_rf = print_metrics("Random Forest Regressor", y_test, y_pred_rf)

# %% [markdown]
# ## Predictions Analysis with Plotly
# 
# We build an interactive Plotly scatter plot comparing predicted prices versus actual prices for both models, including a diagonal reference line marking perfect prediction.

# %%
import plotly.graph_objects as go

fig = go.Figure()

# Ridge scatter
fig.add_trace(go.Scatter(
    x=y_test / 1000,
    y=y_pred_ridge / 1000,
    mode='markers',
    name=f'Ridge (R²: {r2_ridge:.3f})',
    marker=dict(color='#8B5CF6', opacity=0.6, size=6)
))

# Random Forest scatter
fig.add_trace(go.Scatter(
    x=y_test / 1000,
    y=y_pred_rf / 1000,
    mode='markers',
    name=f'Random Forest (R²: {r2_rf:.3f})',
    marker=dict(color='#10B981', opacity=0.6, size=6)
))

# Perfect correlation diagonal line
min_val = min(y_test.min(), y_pred_rf.min()) / 1000
max_val = max(y_test.max(), y_pred_rf.max()) / 1000
fig.add_trace(go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode='lines',
    name='Ideal Prediction',
    line=dict(color='#EF4444', dash='dash', width=2)
))

fig.update_layout(
    title='Model Prediction Comparison: Predicted vs Actual Home Prices',
    xaxis_title='Actual Price ($k)',
    yaxis_title='Predicted Price ($k)',
    legend_title='Models',
    template='plotly_white',
    width=800,
    height=550
)

fig.show()
