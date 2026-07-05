# ---
# title: "Athlete Multi-Dimensional Attribute Radar Profiles"
# description: "Maps multivariate athlete performance attributes. Builds filled polar radar chart overlays using Plotly."
# category: "EDA"
# tags: ["EDA", "Multivariate Analysis", "Radar Chart", "Plotly"]
# date: "2026-07-05"
# metrics:
#   Dimensions: "6"
#   Profiles Compared: "2"
#   Performance Index: "0-100"
# ---

# %% [markdown]
# # Athlete Attribute Radar Analysis
# 
# Radar charts (spider charts) project multivariate data coordinates onto polar axes. This visualization excels at comparing attribute profiles across individuals or models. This notebook defines synthetic attribute metrics (speed, strength, stamina, technique, tactical sense, psychological toughness) for two athlete profiles and constructs an interactive filled radar chart overlay using Plotly.

# %%
import pandas as pd
import plotly.graph_objects as go

# Define performance attributes categories
categories = ['Speed', 'Strength', 'Stamina', 'Technique', 'Tactical', 'Psychological']

# Performance metrics out of 100 for two profiles (e.g. Attacker vs. Midfielder)
attacker_stats = [92, 78, 80, 88, 74, 85]
midfielder_stats = [78, 70, 95, 91, 93, 80]

df_radar = pd.DataFrame({
    'Attribute': categories,
    'Attacker_Profile': attacker_stats,
    'Midfielder_Profile': midfielder_stats
})

df_radar

# %% [markdown]
# ## Close Polar Coordinates
# 
# To ensure the line boundaries close back onto the starting point in polar radar projections, we append the first attribute element value to the end of the arrays.

# %%
# Append starting category to close radar path shapes
radar_categories = categories + [categories[0]]
attacker_closed = attacker_stats + [attacker_stats[0]]
midfielder_closed = midfielder_stats + [midfielder_stats[0]]

# %% [markdown]
# ## Filled Radar Chart Overlay
# 
# Using Plotly Scatterpolar, we draw filled polygonal attribute outlines for both profiles to compare visual attribute shapes.

# %%
fig = go.Figure()

# 1. Attacker Shape
fig.add_trace(go.Scatterpolar(
    r=attacker_closed,
    theta=radar_categories,
    fill='toself',
    name='Attacker Profile',
    fillcolor='rgba(239, 68, 68, 0.3)',  # Red transparent fill
    line=dict(color='#EF4444', width=2)
))

# 2. Midfielder Shape
fig.add_trace(go.Scatterpolar(
    r=midfielder_closed,
    theta=radar_categories,
    fill='toself',
    name='Midfielder Profile',
    fillcolor='rgba(59, 130, 246, 0.3)', # Blue transparent fill
    line=dict(color='#3B82F6', width=2)
))

# Configure polar layout
fig.update_layout(
    title='Multi-Dimensional Athlete Attribute Profiling',
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100],
            gridcolor='#E2E8F0'
        ),
        angularaxis=dict(gridcolor='#E2E8F0')
    ),
    showlegend=True,
    template='plotly_white',
    width=600,
    height=480
)

fig.show()
