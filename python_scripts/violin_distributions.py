# ---
# title: "Multi-Group Violin Probability Density Analysis"
# description: "Models probability density profiles across categorical groups. Generates annotated multi-group Violin plots using Seaborn."
# category: "EDA"
# tags: ["EDA", "Statistical Visualization", "Violin Plot", "Seaborn"]
# date: "2026-07-04"
# metrics:
#   Target Groups: "4"
#   Sample Size: "800"
#   Width Parameter: "0.8"
# ---

# %% [markdown]
# # Multi-Group Violin Probability Density Analysis
# 
# Violin plots combine box plots and kernel density plots to visualize the distribution shape, probability density, and summary statistics of numerical columns across categories. This notebook generates synthetic performance metrics across four corporate divisions (R&D, Sales, Marketing, HR) and visualizes distributions using a customized Seaborn Violin plot.

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Generate synthetic performance scores (800 records across 4 departments)
np.random.seed(60)
n_samples = 800

departments = np.random.choice(['R&D', 'Sales', 'Marketing', 'HR'], size=n_samples, p=[0.3, 0.25, 0.25, 0.2])

# Performance score distributions vary per department (e.g. R&D has high variance)
scores = []
for dept in departments:
    if dept == 'R&D':
        score = np.random.normal(82, 9)
    elif dept == 'Sales':
        score = np.random.normal(76, 7)
    elif dept == 'Marketing':
        score = np.random.normal(79, 6)
    else: # HR
        score = np.random.normal(73, 5)
    scores.append(score)

# Clip scores to [50, 100]
scores = np.clip(scores, 50, 100)

df_perf = pd.DataFrame({
    'Department': departments,
    'Performance_Score': scores
})

df_perf.head(10)

# %% [markdown]
# ## Descriptive Statistics Aggregations
# 
# We calculate the mean, median, standard deviation, and sample counts for each department category to support the visual data profiles.

# %%
df_stats = df_perf.groupby('Department')['Performance_Score'].agg(['count', 'mean', 'median', 'std'])
print("Department Performance Statistics:")
df_stats

# %% [markdown]
# ## Annotated Multi-Group Violin Plot
# 
# Using Seaborn, we visualize the department distributions as a vertical violin plot, overlaying inner quartile lines and adding custom styling.

# %%
plt.figure(figsize=(9, 6))

# Plot violins
sns.violinplot(
    x='Department', 
    y='Performance_Score', 
    data=df_perf, 
    palette='muted',
    inner='quartile', # Show quartiles inside the violins
    linewidth=1.8,
    scale='width'
)

plt.title('Performance Score Probability Density Distributions by Corporate Division', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Corporate Division Department', fontsize=11)
plt.ylabel('Evaluated Performance Score (50-100 scale)', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
