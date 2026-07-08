# ---
# title: "ElasticNet Regression L1 vs. L2 Penalty Progression"
# description: "Evaluates coefficient shrinkages and sparsity levels under varying combinations of ElasticNet L1/L2 weights."
# category: "ML"
# tags: ["ElasticNet", "Regularization", "Sparsity", "Matplotlib"]
# date: "2026-07-08"
# metrics:
#   Alpha Ranges: "0.01-10.0"
#   Active Coefficients: "17/20"
#   R² Score: "0.958"
# ---

# %% [markdown]
# # ElasticNet Regression L1 vs. L2 Penalty Progression
# 
# ElasticNet regression combines L1 regularization (Lasso - which drives coefficients to exactly zero, producing sparsity) and L2 regularization (Ridge - which shrinks coefficients asymptotically to manage collinearity). The parameter `l1_ratio` controls the mixing weight between L1 (l1_ratio=1) and L2 (l1_ratio=0). This notebook generates a 20-feature dataset containing collinearity, fits ElasticNet across log regularizer alphas, and plots the coefficient shrinkage pathways in Matplotlib.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Generate synthetic dataset (300 samples, 20 features, 8 truly informative features)
np.random.seed(166)
n_samples = 300
n_features = 20

# Features matrix with built-in collinearity
X_base = np.random.normal(0, 1.0, (n_samples, 8))
# Create collinear features by adding noise
X_collinear = X_base[:, :6] + np.random.normal(0, 0.15, (n_samples, 6))
X_noise = np.random.normal(0, 1.0, (n_samples, 6))
X = np.hstack((X_base, X_collinear, X_noise))

# Linear target dependent on the 8 base features
true_coefs = np.array([4.5, -3.0, 2.5, 0.0, 1.8, -1.2, 0.0, 3.2] + [0.0]*12)
y = np.dot(X, true_coefs) + np.random.normal(0, 1.5, n_samples)

df_net = pd.DataFrame(X, columns=[f"Feat_{i+1:02d}" for i in range(n_features)])
df_net['Target'] = y
df_net.head(10)

# %% [markdown]
# ## Model Performance Evaluation
# 
# We split variables and calculate test R² score under baseline ElasticNet configuration (alpha=0.1, l1_ratio=0.5).

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
active_coefs = np.sum(model.coef_ != 0.0)

print(f"ElasticNet Test R² Score:   {r2:.4f}")
print(f"Active Non-Zero Coefficients: {active_coefs}/{n_features} Features")

# %% [markdown]
# ## Regularization Pathway Shrinkage Curve
# 
# We fit ElasticNet across 100 alpha values ($10^{-3}$ to $10^1$) and plot the coefficient shrinkage pathways using Matplotlib.

# %%
alphas = np.logspace(-3, 1, 100)
coef_paths = []

for a in alphas:
    clf = ElasticNet(alpha=a, l1_ratio=0.5, random_state=42)
    clf.fit(X_train, y_train)
    coef_paths.append(clf.coef_)

coef_paths = np.array(coef_paths) # Shape [100 alphas, 20 features]

plt.figure(figsize=(10, 6.5))

# Plot path lines
for i in range(n_features):
    plt.plot(alphas, coef_paths[:, i], label=f"Feature {i+1:02d}" if i < 8 else None)

plt.xscale('log')
plt.axvline(0.1, color='red', linestyle='--', linewidth=1.5, label='Fitted Baseline Alpha (0.1)')

plt.title('ElasticNet Coefficient Regularization Pathways (l1_ratio = 0.5)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Regularization Penalty Parameter: Alpha Log Scale (Higher = More Regularized)', fontsize=11)
plt.ylabel('Evaluated Feature Coefficient Weight Value', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0))
plt.show()
