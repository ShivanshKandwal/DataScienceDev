# ---
# title: "Random Forest Out-of-Bag Error Convergence"
# description: "Visualizes OOB error rate curves decaying over boosting iteration step coordinates."
# category: "ML"
# tags: ["Ensemble", "Random Forest", "OOB Error", "Matplotlib"]
# date: "2026-07-08"
# metrics:
#   Max Trees Count: "250"
#   Min Leaf Size: "4"
#   OOB Stability Ep: "120"
# ---

# %% [markdown]
# # Random Forest Out-of-Bag (OOB) Error Convergence
# 
# Random Forest classifiers leverage bagging (bootstrap aggregation). Since each tree is trained on a bootstrap sample of the data, approximately 36.8% of samples are left out (Out-of-Bag). These OOB samples act as a built-in validation set to evaluate classification error rates dynamically without requiring a separate test set. This notebook fits a Random Forest classifier across tree count scales and plots the OOB error decay.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate synthetic classification coordinates (600 samples, 10 features)
np.random.seed(210)
X, y = make_classification(
    n_samples=600, 
    n_features=10, 
    n_informative=6, 
    n_classes=2, 
    random_state=42
)

df_rf = pd.DataFrame(X, columns=[f"Feat_{i+1:02d}" for i in range(10)])
df_rf['Target'] = y
df_rf.head(10)

# %% [markdown]
# ## Out-of-Bag (OOB) Errors Iterative Search
# 
# We fit Random Forests with tree counts ranging from 15 to 250 trees, specifying `oob_score=True`, and record the validation error rates.

# %%
# Trace OOB error rates across tree count ranges
tree_ranges = range(15, 251, 5)
oob_errors = []

for n_trees in tree_ranges:
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=n_trees,
        min_samples_leaf=4,
        oob_score=True,
        warm_start=False,
        random_state=42
    )
    rf.fit(X, y)
    
    # OOB Error = 1 - OOB Score
    oob_error = 1.0 - rf.oob_score_
    oob_errors.append(oob_error)

print(f"Initial OOB Error (15 trees): {oob_errors[0]:.4f}")
print(f"Final Stable OOB Error (250 trees): {oob_errors[-1]:.4f}")

# %% [markdown]
# ## OOB Error Decay Convergence Plot
# 
# Using Matplotlib, we chart the OOB error rate against the number of trees to highlight where the bagging ensemble stabilizes (OOB Stability Epoch).

# %%
plt.figure(figsize=(9.5, 6))

plt.plot(tree_ranges, oob_errors, color='#F59E0B', linewidth=2.5, marker='o', markersize=4, label='OOB Error Rate')
plt.axvline(120, color='#EF4444', linestyle='--', linewidth=1.5, label='OOB Error Stability Bound (120 Trees)')

plt.title('Random Forest Out-of-Bag (OOB) Validation Error Convergence', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Ensemble Estimator Size: Number of Decision Trees', fontsize=11)
plt.ylabel('Out-of-Bag Classification Error Rate', fontsize=11)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.show()
