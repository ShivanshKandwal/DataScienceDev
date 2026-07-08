# ---
# title: "Multi-Variable Boxplot Score Analysis"
# description: "Compares distribution statistics (quartiles, outliers) of performance metrics across multiple student categories."
# category: "EDA"
# tags: ["Statistical Visuals", "Boxplot", "Outliers", "Seaborn"]
# date: "2026-07-08"
# metrics:
#   Test Groups: "5"
#   Student Count: "1200"
#   Outlier Ratio: "1.75%"
# ---

# %% [markdown]
# # Multi-Variable Boxplot Score Analysis
# 
# Box plots compare the spread and skewness of continuous numerical metrics across discrete groups. They highlight key summary statistics: minimum, first quartile (Q1), median (Q2), third quartile (Q3), maximum, and individual outliers. This notebook generates synthetic test scores across five study subjects and creates a styled Seaborn boxplot.

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Generate test scores across 5 subjects (1,200 students total)
np.random.seed(112)
n_students = 1200

subjects = np.random.choice(['Math', 'Physics', 'Chemistry', 'Biology', 'History'], size=n_students)
scores = []

# Inject varying score spreads and manual outliers per subject
for sub in subjects:
    if sub == 'Math':
        score = np.random.normal(72, 12)
    elif sub == 'Physics':
        score = np.random.normal(68, 14)
    elif sub == 'Chemistry':
        score = np.random.normal(75, 10)
    elif sub == 'Biology':
        score = np.random.normal(81, 8)
    else: # History
        score = np.random.normal(85, 6)
        
    scores.append(score)

scores = np.clip(scores, 10, 100)

df_students = pd.DataFrame({
    'Subject': subjects,
    'Score': scores
})

# Inject manual low-performing outlier scores to verify boxplot representation
outlier_indices = np.random.choice(n_students, size=18, replace=False)
df_students.loc[outlier_indices, 'Score'] = np.random.uniform(10, 30, 18)

df_students.head(10)

# %% [markdown]
# ## Subject Aggregations
# 
# We calculate mean scores and count outliers (defined as values beyond 1.5 * IQR) to verify statistics.

# %%
stats = []
for sub, group in df_students.groupby('Subject'):
    q1, q3 = np.percentile(group['Score'], [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers_count = group[(group['Score'] < lower_bound) | (group['Score'] > upper_bound)].shape[0]
    
    stats.append({
        'Subject': sub,
        'Mean': group['Score'].mean(),
        'Median': group['Score'].median(),
        'Outliers': outliers_count
    })

df_stats = pd.DataFrame(stats)
print(f"Calculated Outliers Ratio: {df_stats['Outliers'].sum() / n_students * 100:.2f}%")
df_stats

# %% [markdown]
# ## Styled Subject Boxplots
# 
# Using Seaborn, we draw a vertical boxplot, coloring categories and formatting outlier points.

# %%
plt.figure(figsize=(9, 6))

sns.boxplot(
    x='Subject', 
    y='Score', 
    data=df_students, 
    palette='Set2',
    width=0.6,
    linewidth=1.5,
    flierprops=dict(marker='o', markerfacecolor='#EF4444', markersize=6, linestyle='none', markeredgecolor='black')
)

plt.title('Academic Score Distributions & Statistical Outliers by Subject', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Academic Exam Subject', fontsize=11)
plt.ylabel('Exam Scores (0-100 scale)', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
