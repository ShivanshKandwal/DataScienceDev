# ---
# title: "Partial Least Squares (PLS) Regression Predictions"
# description: "Decomposes collinear predictors coordinates into latent variables projections to forecast target outputs."
# category: "ML"
# tags: ["PLS", "Collinearity", "Regression", "Scikit-Learn"]
# date: "2026-07-08"
# metrics:
#   Predictors: "30"
#   Latent Variables: "3"
#   Test R²: "0.978"
# ---

# %% [markdown]
# # Partial Least Squares (PLS) Regression
# 
# Ordinary Least Squares (OLS) regression fails when predictors are highly collinear or outnumber samples. Partial Least Squares (PLS) regression projects both predictor variables ($X$) and target responses ($Y$) onto lower-dimensional latent spaces to maximize their covariance before fitting. This notebook generates 30 collinear features and trains a PLS Regressor using Scikit-Learn.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Generate synthetic collinear data (300 samples, 30 features, collinear blocks)
np.random.seed(200)
n_samples = 300
n_features = 30

# Base latent factors generating the collinearity
latent_base = np.random.normal(0, 1.0, (n_samples, 3))

# Expand 3 latent factors to 30 features with noise
X = np.zeros((n_samples, n_features))
for i in range(n_features):
    # Features correlate with different latent combinations
    factor_weight = np.random.uniform(-1, 1, 3)
    X[:, i] = np.dot(latent_base, factor_weight) + np.random.normal(0, 0.25, n_samples)

# Target depends linearly on latent factors
y = 2.5 * latent_base[:, 0] - 1.8 * latent_base[:, 1] + 1.2 * latent_base[:, 2] + np.random.normal(0, 0.4, n_samples)

df_pls = pd.DataFrame(X, columns=[f"Pred_{i+1:02d}" for i in range(n_features)])
df_pls['Target'] = y
df_pls.head(10)

# %% [markdown]
# ## PLS Regression Pipeline Fitting
# 
# We split variables into 80/20 train/test segments, fit a PLS Regressor extracting 3 latent variables, and compute target prediction metrics.

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train PLS Regression with 3 latent components
pls = PLSRegression(n_components=3)
pls.fit(X_train, y_train)

# Predict targets
y_pred = pls.predict(X_test).ravel()

# Evaluate scores
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"PLS Regression Test R² Score: {r2:.4f}")
print(f"PLS Regression Test MSE:      {mse:.4f}")

# %% [markdown]
# ## PLS Latent Projection & Predictions Alignment
# 
# We visualize actual vs. predicted values in a scatter plot using Matplotlib, showing how the 3 latent variables track the regression target.

# %%
plt.figure(figsize=(9, 6.5))

# Plot scatter actual vs predicted
plt.scatter(y_test, y_pred, c='#8B5CF6', edgecolors='k', alpha=0.75, s=40, label='Test Samples')

# Draw ideal 45-degree diagonal line
ideal_min, ideal_max = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
plt.plot([ideal_min, ideal_max], [ideal_min, ideal_max], color='#EF4444', linestyle='--', linewidth=1.8, label='Ideal Alignment')

plt.title('PLS Regression Predictions: Latent Projection Performance Mapping', fontsize=13, fontweight='bold')
plt.xlabel('Ground Truth Target Values (y_test)', fontsize=11)
plt.ylabel('Model Predicted Values (y_pred)', fontsize=11)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.show()
