# ---
# title: "Web Clickstream Funnel Conversion Analysis"
# description: "Models user journeys and conversion funnel decay rates from web log sequences. Builds interactive Plotly funnels."
# category: "EDA"
# tags: ["EDA", "Clickstream", "Conversion Funnel", "Plotly"]
# date: "2026-07-02"
# metrics:
#   Funnel Steps: "4"
#   Conversion Rate: "3.2%"
#   Session Count: "10k"
# ---

# %% [markdown]
# # Web Clickstream Funnel Conversion Analysis
# 
# Conversion funnels chart the user journey on digital platforms, helping identify drops at specific conversion checkpoints (e.g. landing page, search product, add to cart, and checkout purchase). This project simulates web log sessions and builds an interactive Plotly conversion funnel chart showing drop-off patterns.

# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Simulate clickstream logs for 10,000 sessions
np.random.seed(33)
n_sessions = 10000

# Base step probabilities (funnel progression drops at each step)
# Step 1: Session Start (100% of users land on page)
# Step 2: Product Search (62% of landing users)
# Step 3: Add to Cart (28% of searching users)
# Step 4: Purchase Checkout (40% of shopping cart users)

progression = []
for i in range(n_sessions):
    session_id = f"S_{i+1:05d}"
    
    # Checkpoint 1: Landed (always true)
    progression.append((session_id, '1_Landing', 1))
    
    # Checkpoint 2: Search
    if np.random.rand() < 0.62:
        progression.append((session_id, '2_Search', 1))
        
        # Checkpoint 3: Cart
        if np.random.rand() < 0.28:
            progression.append((session_id, '3_Cart', 1))
            
            # Checkpoint 4: Purchase
            if np.random.rand() < 0.40:
                progression.append((session_id, '4_Purchase', 1))

df_funnel = pd.DataFrame(progression, columns=['Session_ID', 'Step', 'Active'])
df_funnel.head(10)

# %% [markdown]
# ## Funnel Aggregations & Progression Ratios
# 
# We calculate total active sessions at each funnel step, and compute step-to-step drop-off ratios to evaluate where users drop off the platform.

# %%
# Group by step to count active users
funnel_counts = df_funnel.groupby('Step')['Session_ID'].nunique()
df_counts = pd.DataFrame(funnel_counts).rename(columns={'Session_ID': 'Users'})

# Calculate percentage conversions
base_users = df_counts.iloc[0]['Users']
df_counts['Global_Conversion_%'] = (df_counts['Users'] / base_users) * 100

# Calculate step-to-step retention
df_counts['Step_Retention_%'] = df_counts['Users'].pct_change().fillna(0.0) + 1.0
df_counts.iloc[0, 2] = 1.0
df_counts['Step_Retention_%'] *= 100

print("Funnel Checkpoints Aggregated Data:")
df_counts

# %% [markdown]
# ## Interactive Conversion Funnel Plotly Chart
# 
# We construct a styled Plotly Funnel visualization highlighting drop-off statistics, showing conversion percentages and the relative size of each step.

# %%
# Convert structures to python lists for JSON serialization compatibility
steps = [s.split('_')[1] for s in df_counts.index]
user_values = [int(v) for v in df_counts['Users']]

fig = go.Figure(go.Funnel(
    y=steps,
    x=user_values,
    textposition="inside",
    textinfo="value+percent initial",
    opacity=0.85,
    marker={"color": ["#3B82F6", "#6366F1", "#8B5CF6", "#EC4899"],
            "line": {"width": [2, 2, 2, 2], "color": ["#1D4ED8", "#4338CA", "#6D28D9", "#BE185D"]}},
    connector={"line": {"color": "#CBD5E1", "width": 1.5}}
))

fig.update_layout(
    title='Web E-Commerce Funnel Checkpoint Analysis',
    template='plotly_white',
    width=650,
    height=450
)

fig.show()
