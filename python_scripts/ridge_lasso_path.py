# ---
# title: "Ridge vs. Lasso Coefficient Shrinkage Paths"
# description: "Plots Lasso (L1) zeroing paths against Ridge (L2) asymptotic decays across log regularizer alphas."
# category: "ML"
# tags: ["Regularization", "Ridge", "Lasso", "Matplotlib"]
# date: "2026-07-08"
# metrics:
#   Features: "15"
#   Alpha Range: "10^-4 to 10^3"
#   L1 Sparsity Count: "6"
# ---

# %% [markdown]
# # Ridge vs. Lasso Coefficient Shrinkage Paths
# 
# L1 regularization (Lasso) and L2 regularization (Ridge) reduce overfitting by adding penalty constraints to linear regressions:
# - **Lasso (L1):** Adds absolute weights sum penalty ($\sum |\beta_j|$), driving non-informative coefficients to exactly zero (enables feature selection/sparsity).
# - **Ridge (L2):** Adds squared weights sum penalty ($\sum \beta_j^2$), shrinking coefficients asymptotically toward zero.
# This notebook generates 15 features, fits both Ridge and Lasso models, and plots side-by-side shrinkage path subplots.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso

# Generate synthetic linear regression data (100 samples, 15 features, 6 informative features)
np.random.seed(190)
n_samples = 100
n_features = 15

X = np.random.normal(0, 1.0, (n_samples, n_features))
# Set only 6 features to be active
true_coefs = np.array([3.5, -2.5, 1.8, -1.2, 0.8, -0.5] + [0.0]*9)
y = np.dot(X, true_coefs) + np.random.normal(0, 0.5, n_samples)

df_reg = pd.DataFrame(X, columns=[f"Feat_{i+1:02d}" for i in range(n_features)])
df_reg['Target'] = y
df_reg.head(10)

# %% [markdown]
# ## Regularization Paths Fitting
# 
# We fit Ridge and Lasso regressors across a log regularizer alpha array ranging from $10^{-4}$ to $10^3$.

# %%
alphas = np.logspace(-4, 3, 100)

ridge_coefs = []
lasso_coefs = []

for a in alphas:
    # Ridge L2
    ridge = Ridge(alpha=a)
    ridge.fit(X, y)
    ridge_coefs.append(ridge.coef_)
    
    # Lasso L1 (max_iter increased for convergence stability)
    lasso = Lasso(alpha=a, max_iter=5000)
    lasso.fit(X, y)
    lasso_coefs.append(lasso.coef_)

ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)

print(f"Computed Ridge Coefficient Matrix: {ridge_coefs.shape}")
print(f"Computed Lasso Coefficient Matrix: {lasso_coefs.shape}")

# %% [markdown]
# ## Coefficient Shrinkage Paths Comparison
# 
# Using a 1x2 Matplotlib subplot grid, we compare Ridge's asymptotic decay against Lasso's sparse zeroing behavior across log alphas.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 1. Ridge Shrinkage Paths (L2)
for i in range(n_features):
    ax1.plot(alphas, ridge_coefs[:, i], label=f"Feat {i+1:02d}" if i < 6 else None)
ax1.set_xscale('log')
ax1.set_title('Ridge Regularization Paths (L2 Penalty - Asymptotic Decay)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Regularization Parameter Alpha (Log Scale)')
ax1.set_ylabel('Coefficient Weight')
ax1.grid(True, linestyle=':', alpha=0.5)

# 2. Lasso Shrinkage Paths (L1)
for i in range(n_features):
    ax2.plot(alphas, lasso_coefs[:, i], label=f"Feat {i+1:02d}" if i < 6 else None)
ax2.set_xscale('log')
ax2.set_title('Lasso Regularization Paths (L1 Penalty - Sparse Feature Zeroing)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Regularization Parameter Alpha (Log Scale)')
ax2.set_ylabel('Coefficient Weight')
ax2.grid(True, linestyle=':', alpha=0.5)

plt.suptitle('Linear Model Regularization Limits: Ridge (L2) vs. Lasso (L1) Pathways', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()
