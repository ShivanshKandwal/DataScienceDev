# ---
# title: "Demographic Population Pyramid Analysis"
# description: "Models demographic distributions split by age cohorts and gender. Generates mirrored horizontal bar charts using Plotly."
# category: "EDA"
# tags: ["EDA", "Demographics", "Population Pyramid", "Plotly"]
# date: "2026-07-04"
# metrics:
#   Age Groups: "10"
#   Cohort Width: "10 Years"
#   Total Pop: "100k"
# ---

# %% [markdown]
# # Demographic Population Pyramid Analysis
# 
# Population pyramids are mirrored horizontal bar charts that visualize the distribution of age groups and genders within a population. This structural view helps demographers study growth rates and aging trends. This notebook simulates demographic census records, divides them into 10-year age cohorts, and visualizes the structure using an interactive Plotly population pyramid.

# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Generate synthetic census population profiles (100,000 records)
np.random.seed(50)
n_population = 100000

# Generate ages with a realistic demographic distribution (more young people, fewer elderly)
ages = np.random.exponential(scale=38, size=n_population).clip(0, 95)
genders = np.random.choice(['Male', 'Female'], size=n_population, p=[0.495, 0.505])

df_census = pd.DataFrame({
    'Age': ages,
    'Gender': genders
})

# Define 10-year age bins
bins = list(range(0, 101, 10))
labels = [f"{i}-{i+9}" for i in range(0, 90, 10)] + ["90+"]

df_census['Age_Group'] = pd.cut(df_census['Age'], bins=bins, labels=labels, right=False)
df_census.head(10)

# %% [markdown]
# ## Demographic Aggregations
# 
# We calculate total population counts for each combination of age group and gender, and express males as negative values so they plot leftward on the horizontal axis.

# %%
# Group and count population
df_counts = df_census.groupby(['Age_Group', 'Gender'], observed=False).size().unstack(fill_value=0)

# Convert to percentage of total population for scaling
total_pop = df_counts.sum().sum()
df_pct = (df_counts / total_pop) * 100

# Make male counts negative for leftward plotting
df_pct['Male_Negative'] = -df_pct['Male']

print("Age Cohort Distribution Percentages (%):")
df_pct

# %% [markdown]
# ## Mirrored Population Pyramid Plot
# 
# Using Plotly, we render the population pyramid by overlaying positive female values (right side) and negative male values (left side) along a shared horizontal axis, applying custom styling and formatting.

# %%
# Convert structures to standard lists for JSON serialization compatibility
age_groups = list(df_pct.index)
male_pcts = [float(v) for v in df_pct['Male_Negative']]
female_pcts = [float(v) for v in df_pct['Female']]

fig = go.Figure()

# 1. Male Bar Chart Trace (Left Side)
fig.add_trace(go.Bar(
    y=age_groups,
    x=male_pcts,
    orientation='h',
    name='Male',
    marker_color='#3B82F6',
    hoverinfo='text',
    hovertext=[f"Age {grp}: {abs(val):.2f}% Male" for grp, val in zip(age_groups, male_pcts)]
))

# 2. Female Bar Chart Trace (Right Side)
fig.add_trace(go.Bar(
    y=age_groups,
    x=female_pcts,
    orientation='h',
    name='Female',
    marker_color='#EC4899',
    hoverinfo='text',
    hovertext=[f"Age {grp}: {val:.2f}% Female" for grp, val in zip(age_groups, female_pcts)]
))

# Configure horizontal axis styling
fig.update_layout(
    title='Demographic Population Pyramid (% of Total Population)',
    barmode='relative',
    xaxis=dict(
        title='Percentage of Population (%)',
        tickvals=[-6, -4, -2, 0, 2, 4, 6],
        ticktext=['6%', '4%', '2%', '0%', '2%', '4%', '6%']
    ),
    yaxis=dict(title='Age Cohort Group'),
    template='plotly_white',
    width=650,
    height=480,
    legend_title='Gender'
)

fig.show()
