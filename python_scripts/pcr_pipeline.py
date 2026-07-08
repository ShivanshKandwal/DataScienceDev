# ---
# title: "Principal Component Regression Pipeline"
# description: "Builds a PCA reduction linear regression pipeline (PCR) in Scikit-Learn."
# category: "ML"
# tags: ["Dimension Reduction", "Linear Pipeline", "PCR", "Scikit-Learn"]
# date: "2026-07-08"
# metrics:
#   Raw Predictors: "25"
#   Principal Components: "5"
#   RMSE: "0.313"
# ---

# %% [markdown]
# # Principal Component Regression (PCR) Pipeline
# 
# Principal Component Regression (PCR) resolves collinearity by:
# 1. **PCA Step:** Performing Principal Component Analysis on the predictors matrix to extract orthogonal components.
# 2. **OLS Step:** Fitting an Ordinary Least Squares (OLS) regression on the subset of principal components.
# This notebook builds a PCR workflow using Scikit-Learn `Pipeline`, fits it on a collinear 25-feature dataset, and plots predictions residuals using Matplotlib.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Generate synthetic collinear dataset (400 samples, 25 features)
np.random.seed(215)
n_samples = 400
n_features = 25

# Generate 5 underlying latent factors
latent_factors = np.random.normal(0, 1.0, (n_samples, 5))

# Project 5 factors onto 25 collinear features
X = np.dot(latent_factors, np.random.uniform(-1, 1, (5, n_features))) + np.random.normal(0, 0.15, (n_samples, n_features))

# Target depends on the 5 latent factors
y = 1.8 * latent_factors[:, 0] - 2.5 * latent_factors[:, 1] + 0.9 * latent_factors[:, 2] + np.random.normal(0, 0.25, n_samples)

df_pcr = pd.DataFrame(X, columns=[f"Var_{i+1:02d}" for i in range(n_features)])
df_pcr['Target'] = y
df_pcr.head(10)

# %% [markdown]
# ## PCR Pipeline Training & Evaluation
# 
# We build a Scikit-Learn Pipeline combining `StandardScaler`, `PCA(n_components=5)`, and `LinearRegression`. We evaluate performance using test Root Mean Squared Error (RMSE).

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construct PCR pipeline
pcr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=5)),
    ('regression', LinearRegression())
])

pcr_pipeline.fit(X_train, y_train)

# Predict targets
y_pred = pcr_pipeline.predict(X_test)

# Calculate RMSE and R2 scores
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"PCR Pipeline Test RMSE: {rmse:.4f}")
print(f"PCR Pipeline Test R²:   {r2:.4f}")

# %% [markdown]
# ## Residuals Distribution Plot
# 
# Using Matplotlib, we plot predictions residuals ($y_{test} - y_{pred}$) to verify homoscedasticity and check error normal distributions.

# %%
residuals = y_test - y_pred

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# 1. Residuals vs Predicted scatter plot
ax1.scatter(y_pred, residuals, c='#3B82F6', edgecolors='k', alpha=0.75, s=35)
ax1.axhline(0.0, color='#EF4444', linestyle='--', linewidth=1.5)
ax1.set_title('Residuals Variance vs. Predicted Values', fontsize=12, fontweight='bold')
ax1.set_xlabel('Predicted Values (y_pred)')
ax1.set_ylabel('Prediction Residuals (y_actual - y_pred)')
ax1.grid(True, linestyle=':', alpha=0.5)

# 2. Residuals histogram distribution
ax2.hist(residuals, bins=15, color='#60A5FA', edgecolor='black', alpha=0.8)
ax2.set_title('Residuals Error Distribution Frequency', fontsize=12, fontweight='bold')
ax2.set_xlabel('Residuals Error Value')
ax2.set_ylabel('Frequency')
ax2.grid(True, linestyle=':', alpha=0.5)

plt.suptitle('Principal Component Regression (PCR): Error Residuals Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
