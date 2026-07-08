# ---
# title: "Multi-Output Chain Regression Predictions"
# description: "Trains chain regressors to predict correlated target dimensions concurrently (e.g. multiple air pollutant metrics)."
# category: "ML"
# tags: ["Regression", "Multi-Output", "Chain Estimator", "Scikit-Learn"]
# date: "2026-07-08"
# metrics:
#   Target Variables: "3"
#   Feature Vectors: "8"
#   Mean R²: "0.543"
# ---

# %% [markdown]
# # Multi-Output Chain Regression
# 
# Standard multi-output regressors forecast target variables independently, ignoring target dependencies. **Regressor Chains** sequence targets in a chain structure: subsequent targets are predicted using the input features *plus* the predictions of previous targets, preserving target correlations. This notebook trains a Regressor Chain to predict three correlated air pollutant levels and visualizes prediction sequences.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.multioutput import RegressorChain
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Generate synthetic air quality metrics (400 samples, 8 features, 3 targets)
np.random.seed(180)
n_samples = 400
n_features = 8

# Input features (meteorological factors: Temp, Humidity, Wind, etc.)
X = np.random.normal(20, 5, (n_samples, n_features))

# Correlated targets (PM2.5, PM10, NO2)
# PM10 is dependent on PM2.5, NO2 is dependent on both
target_1 = 0.6 * X[:, 0] - 0.3 * X[:, 2] + np.random.normal(15, 3, n_samples) # PM2.5
target_2 = 0.8 * target_1 + 0.2 * X[:, 1] + np.random.normal(5, 2, n_samples) # PM10
target_3 = 0.4 * target_1 + 0.5 * target_2 - 0.2 * X[:, 3] + np.random.normal(10, 1.5, n_samples) # NO2

Y = np.column_stack((target_1, target_2, target_3))

df_features = pd.DataFrame(X, columns=[f"Met_Feat_{i+1}" for i in range(n_features)])
df_targets = pd.DataFrame(Y, columns=['PM2.5', 'PM10', 'NO2'])
pd.concat([df_features.head(10), df_targets.head(10)], axis=1)

# %% [markdown]
# ## Regressor Chain Training & Target Evaluation
# 
# We split our air quality data into 80/20 train/test segments, fit a Regressor Chain containing sequential Ridge Estimators, and calculate R² metrics across the 3 targets.

# %%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build Regressor Chain using Ridge base estimator
base_estimator = Ridge(alpha=1.0)
chain = RegressorChain(base_estimator, order=[0, 1, 2], random_state=42)
chain.fit(X_train, y_train)

# Predict targets
y_pred = chain.predict(X_test)

# Calculate R² scores per target
r2_scores = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(3)]
mean_r2 = np.mean(r2_scores)

print("Target-Specific R² Scores:")
print(f"Target 1 (PM2.5): {r2_scores[0]:.4f}")
print(f"Target 2 (PM10):  {r2_scores[1]:.4f}")
print(f"Target 3 (NO2):   {r2_scores[2]:.4f}")
print(f"\nRegressor Chain Mean R² Score: {mean_r2:.4f}")

# %% [markdown]
# ## Target Predictions Sequence Plot
# 
# We plot actual values vs. Regressor Chain predicted curves for the first 40 test samples using Matplotlib to demonstrate sequence correlation tracking.

# %%
plt.figure(figsize=(10, 6.5))

# Plot actual vs predicted sequences for a subset of samples
plot_samples = 40
x_samples = range(plot_samples)

colors = ['#EF4444', '#3B82F6', '#10B981']
targets_names = ['PM2.5', 'PM10', 'NO2']

for i in range(3):
    # Actual targets (solid lines)
    plt.plot(x_samples, y_test[:plot_samples, i], color=colors[i], linestyle='-', alpha=0.4, linewidth=1.5)
    # Predicted targets (dashed lines with markers)
    plt.plot(
        x_samples, 
        y_pred[:plot_samples, i], 
        color=colors[i], 
        linestyle='--', 
        marker='o', 
        markersize=4.5, 
        label=f'{targets_names[i]} (R²: {r2_scores[i]:.2f})',
        linewidth=1.8
    )

plt.title('Regressor Chain Predictions Tracking: Air Quality Multi-Output Estimator', fontsize=13, fontweight='bold')
plt.xlabel('Test Dataset Sample Index', fontsize=11)
plt.ylabel('Evaluated Pollutant Density Level (ug/m³)', fontsize=11)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.show()
