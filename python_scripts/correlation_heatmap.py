# ---
# title: "Statistical Correlation Matrix Heatmap"
# description: "Models Pearson correlation matrices across synthetic variables. Renders annotated correlation heatmaps using Seaborn."
# category: "EDA"
# tags: ["EDA", "Correlation Matrix", "Heatmap", "Seaborn"]
# date: "2026-07-06"
# metrics:
#   Features: "6"
#   Sample Size: "500"
#   Max Correlation: "0.85"
# ---

# %% [markdown]
# # Statistical Correlation Matrix Heatmap
# 
# Correlation matrix heatmaps visualize the linear relationships (Pearson correlation coefficients) between multiple continuous numerical columns. Finding highly correlated variables is a crucial step in exploratory data analysis and feature engineering. This notebook generates synthetic variables with embedded correlations and plots the correlation matrix as a styled Seaborn heatmap.

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Generate correlated synthetic features (500 samples, 6 columns)
np.random.seed(88)
n_samples = 500

# Base independent features
feat_1 = np.random.normal(10, 2, n_samples)
feat_2 = np.random.normal(50, 10, n_samples)
feat_3 = np.random.normal(0, 1, n_samples)

# Dependent correlated features
feat_4 = 0.8 * feat_1 + np.random.normal(0, 0.5, n_samples) # Strong positive correlation with feat_1
feat_5 = -0.75 * feat_2 + np.random.normal(0, 3.0, n_samples) # Strong negative correlation with feat_2
feat_6 = 0.3 * feat_1 - 0.4 * feat_3 + np.random.normal(0, 0.2, n_samples)

df_corr = pd.DataFrame({
    'Income': feat_1,
    'Age': feat_2,
    'Education_Yrs': feat_3,
    'Savings': feat_4,
    'Debt': feat_5,
    'Spending_Score': feat_6
})

df_corr.head(10)

# %% [markdown]
# ## Pearson Correlation Matrix Calculation
# 
# We calculate the symmetric Pearson correlation coefficients matrix across all features.

# %%
corr_matrix = df_corr.corr()
print("Pearson Correlation Coefficients Matrix:")
corr_matrix

# %% [markdown]
# ## Styled Seaborn Correlation Heatmap
# 
# Using Seaborn, we visualize the matrix as an annotated heatmap, using a diverging palette and adding a mask to hide the redundant upper triangle.

# %%
plt.figure(figsize=(9, 7))

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Draw correlation heatmap
sns.heatmap(
    corr_matrix, 
    mask=mask,
    annot=True, 
    fmt=".2f", 
    cmap='coolwarm', 
    vmin=-1.0, 
    vmax=1.0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)

plt.title('Correlation Coefficient Heatmap Matrix (Pearson r)', fontsize=13, fontweight='bold', pad=15)
plt.show()
