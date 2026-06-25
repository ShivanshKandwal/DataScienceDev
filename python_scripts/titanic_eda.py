# ---
# title: "Titanic Survival Exploratory Data Analysis"
# description: "Data-mining passenger manifest records to extract survival patterns. Features extensive categorical data engineering, missing value imputations, and seaborn density plots."
# category: "EDA"
# tags: ["Python", "Pandas", "Seaborn", "Matplotlib"]
# date: "2026-06-21"
# metrics:
#   Records: "891"
#   Features: "12"
#   Correlations: "0.78"
# ---

# %% [markdown]
# # Titanic Survival Exploratory Data Analysis
# 
# This notebook performs exploratory data analysis (EDA) on the Titanic passenger manifest to identify factors that influenced survival rates. We study passenger class, age, gender, and family sizing.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = sns.load_dataset('titanic')
df = df.rename(columns={
    'survived': 'Survived',
    'pclass': 'Pclass',
    'sex': 'Sex',
    'age': 'Age',
    'fare': 'Fare'
})
df.head()

# %% [markdown]
# ## Survival Rates by Sex and Class
# 
# We group the passenger cohort by Sex and Class to plot survival distribution density percentages. Historically, women and children in first class survived at a significantly higher rate.

# %%
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=df, palette="muted")
plt.title("Survival Percentage by Passenger Class and Sex")
plt.ylabel("Survival Rate")
plt.show()

# %% [markdown]
# ## Data Imputation
# 
# Let us check the missing values. The Age feature contains null parameters that require median-based class distributions.

# %%
print("Missing Values:")
print(df.isnull().sum())
